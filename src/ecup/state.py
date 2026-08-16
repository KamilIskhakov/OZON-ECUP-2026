"""Динамическое состояние пользователя: экспоненциальная память и байесовские фильтры.

Прямоугольные окна (7/14/30/…/300 дней) описывают уровень, но не направление:
покупка 29 дней назад весит столько же, сколько вчерашняя, а 31 день назад —
уже ноль. Здесь память затухает плавно, и появляется величина, которую окно
выразить не может: текущая интенсивность **относительно собственного**
долгосрочного уровня пользователя.

Замер на одном якоре: 12 недельных лагов (75 признаков) дают +0.0005 к shape,
EWMA (25 признаков) — +0.0021. Прямоугольные лаги дублируют уже имеющиеся окна,
экспоненциальные — нет.

Байесовские фильтры раскладывают вероятность покупки на два механизма:

    P(покупка в день) = q · c,    q = P(активен),  c = P(покупка | активен)

`days_since_purchase` эти состояния путает: «перестал заходить» и «заходит,
но перестал конвертироваться» дают одинаковый recency, а means они разное.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .features import EPS, RECENCY_NEVER

# Периоды полураспада ≈ 6.6, 23 и 138 дней: короткая, средняя и длинная память.
DELTAS: tuple[tuple[str, float], ...] = (("fast", 0.90), ("mid", 0.97), ("slow", 0.995))
CHANNELS: tuple[tuple[str, str], ...] = (
    ("gmv", "gmv"), ("to_ord", "ord"), ("to_cart", "cart"), ("searches", "srch"),
)
PRIOR_STRENGTH = 3.0      # псевдонаблюдений в бета-приоре


def _decay(anchor: int, delta: float) -> pl.Expr:
    return pl.lit(float(delta)) ** (anchor - pl.col("d")).cast(pl.Float64)


def _total_mass(anchor: int, delta: float, span: int) -> float:
    """Суммарный вес всех дней окна, включая дни без строк.

    Считается аналитически как геометрическая сумма: отсутствующие дни в
    панели не представлены, но для знаменателя «сколько дней вообще прошло»
    они нужны.
    """
    return float((1.0 - delta ** span) / (1.0 - delta))


def ewma_features(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    span: int = 365,
) -> pl.DataFrame:
    """Экспоненциально взвешенные интенсивности и разности быстрых и медленных."""
    h = df.filter(pl.col("user_id").is_in(users)
                  & pl.col("d").is_between(anchor - span + 1, anchor))
    ex: list[pl.Expr] = []
    for nm, dd in DELTAS:
        w = _decay(anchor, dd)
        # умножение на (1-δ) переводит сумму в дневную интенсивность,
        # иначе медленный фильтр всегда крупнее быстрого просто по масштабу
        ex += [((pl.col(c) * w).sum() * (1 - dd)).log1p().alias(f"ewm_{n}_{nm}")
               for c, n in CHANNELS]
        ex.append((w.sum() * (1 - dd)).alias(f"ewm_act_{nm}"))
    f = h.group_by("user_id").agg(ex)
    f = (pl.DataFrame({"user_id": users}).join(f, on="user_id", how="left")
           .with_columns(pl.exclude("user_id").fill_null(0.0)))

    for _, n in (*CHANNELS, ("act", "act")):
        f = f.with_columns(
            (pl.col(f"ewm_{n}_fast") - pl.col(f"ewm_{n}_slow")).alias(f"ewm_{n}_fs"),
            (pl.col(f"ewm_{n}_mid") - pl.col(f"ewm_{n}_slow")).alias(f"ewm_{n}_ms"),
        )
    return f.sort("user_id")


def bayes_filters(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    span: int = 365,
) -> pl.DataFrame:
    """Дисконтированные бета-бернулли фильтры активности и конверсии.

    Каждый день добавляет массу в успехи или неудачи, вся накопленная масса
    домножается на δ — так старая история забывается плавно, а не обрубается
    границей окна.
    """
    h = df.filter(pl.col("user_id").is_in(users)
                  & pl.col("d").is_between(anchor - span + 1, anchor))
    ex: list[pl.Expr] = []
    for nm, dd in DELTAS:
        w = _decay(anchor, dd)
        buy = pl.col("gmv") > 0
        ex += [
            w.sum().alias(f"_act_{nm}"),                              # масса активных дней
            w.filter(buy).sum().alias(f"_buy_{nm}"),                  # масса дней с покупкой
            (pl.col("gmv").log1p() * w).filter(buy).sum().alias(f"_amt_{nm}"),
        ]
    f = h.group_by("user_id").agg(ex)
    f = (pl.DataFrame({"user_id": users}).join(f, on="user_id", how="left")
           .with_columns(pl.exclude("user_id").fill_null(0.0)))

    # опорное значение суммы покупки для сжатия редко покупающих
    amt_prior = float(
        h.filter(pl.col("gmv") > 0)["gmv"].log1p().mean() or 0.0
    )

    for nm, dd in DELTAS:
        total = _total_mass(anchor, dd, span)
        a, b, s = pl.col(f"_act_{nm}"), pl.col(f"_buy_{nm}"), pl.col(f"_amt_{nm}")
        f = f.with_columns(
            # q = P(активен): успехи — активные дни, знаменатель — все дни окна
            ((a + PRIOR_STRENGTH * 0.5) / (total + PRIOR_STRENGTH)).alias(f"q_active_{nm}"),
            # c = P(покупка | активен): знаменатель — только активные дни
            ((b + PRIOR_STRENGTH * 0.5) / (a + PRIOR_STRENGTH)).alias(f"c_conv_{nm}"),
            # характерная сумма покупки, сжатая к общему уровню
            ((s + PRIOR_STRENGTH * amt_prior) / (b + PRIOR_STRENGTH)).alias(f"amt_{nm}"),
        )
    f = f.drop([c for c in f.columns if c.startswith("_")])

    for base in ("q_active", "c_conv", "amt"):
        f = f.with_columns(
            (pl.col(f"{base}_fast") - pl.col(f"{base}_slow")).alias(f"{base}_fs"),
            (pl.col(f"{base}_mid") - pl.col(f"{base}_slow")).alias(f"{base}_ms"),
        )
    # дневная вероятность покупки как произведение двух механизмов
    for nm, _ in DELTAS:
        f = f.with_columns(
            (pl.col(f"q_active_{nm}") * pl.col(f"c_conv_{nm}")).alias(f"p_buy_{nm}"))
    f = f.with_columns(
        (pl.col("p_buy_fast") - pl.col("p_buy_slow")).alias("p_buy_fs"),
        # ожидаемый вклад за 30 дней: интенсивность × горизонт × характерная сумма
        (pl.col("p_buy_fast") * 30.0 * pl.col("amt_fast")).alias("exp_z30_fast"),
        (pl.col("p_buy_slow") * 30.0 * pl.col("amt_slow")).alias("exp_z30_slow"),
    )
    return f.sort("user_id")


def state_features(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    span: int = 365,
    with_ewma: bool = True,
    with_bayes: bool = True,
    with_rhythm: bool = True,
    with_phantom: bool = True,
) -> pl.DataFrame:
    """Полный блок признаков состояния."""
    out = pl.DataFrame({"user_id": users})
    if with_ewma:
        out = out.join(ewma_features(df, anchor, users, span), on="user_id", how="left")
    if with_bayes:
        out = out.join(bayes_filters(df, anchor, users, span), on="user_id", how="left")
    if with_rhythm:
        out = out.join(purchase_rhythm(df, anchor, users, span), on="user_id", how="left")
    if with_phantom:
        out = out.join(phantom_recency(df, anchor, users, span), on="user_id", how="left")
    return out.sort("user_id")


def purchase_rhythm(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    span: int = 365,
) -> pl.DataFrame:
    """Ритм ПОКУПОК, а не визитов.

    BTYD-lite в features.py нормирует recency на медианный интервал между
    визитами. Но по важности признаков видно, что сигнал сидит в покупках:
    `p_buy_slow` забирает больше половины гейна классификатора. Значит и ритм
    надо мерить по покупкам — интервалы между днями с gmv > 0, а не между
    заходами на площадку.

    Ключевой признак — `r_gmv_over_gap`: сколько собственных покупательских
    циклов пользователь молчит. 28 дней тишины у того, кто покупает раз
    в неделю, и у того, кто покупает раз в два месяца, означают разное.
    """
    h = (
        df.filter(pl.col("user_id").is_in(users)
                  & pl.col("d").is_between(anchor - span + 1, anchor)
                  & (pl.col("gmv") > 0))
          .select(["user_id", "d", "gmv"])
          .sort(["user_id", "d"])
    )
    gaps = (
        h.with_columns(gap=(pl.col("d") - pl.col("d").shift(1).over("user_id")))
         .drop_nulls("gap")
    )
    g = gaps.group_by("user_id").agg(
        buy_gap_mean=pl.col("gap").mean(),
        buy_gap_median=pl.col("gap").median(),
        buy_gap_std=pl.col("gap").std(),
        buy_gap_max=pl.col("gap").max(),
        buy_gap_last=pl.col("gap").last(),
        buy_gap_last3=pl.col("gap").tail(3).mean(),
    )
    a = h.group_by("user_id").agg(
        n_buy_days=pl.len(),
        first_buy_ago=(anchor - pl.col("d").min()),
        last_buy_ago=(anchor - pl.col("d").max()),
        buy_gmv_mean=pl.col("gmv").log1p().mean(),
        buy_gmv_std=pl.col("gmv").log1p().std(),
        buy_gmv_max=pl.col("gmv").log1p().max(),
        buy_gmv_last=pl.col("gmv").log1p().last(),
        buy_gmv_last3=pl.col("gmv").log1p().tail(3).mean(),
    )
    f = (
        pl.DataFrame({"user_id": users})
          .join(a, on="user_id", how="left")
          .join(g, on="user_id", how="left")
          .with_columns(pl.exclude("user_id").fill_null(0.0))
    )
    return f.with_columns(
        # молчание в собственных покупательских циклах
        (pl.col("last_buy_ago") / (pl.col("buy_gap_median") + 1.0)).alias("r_gmv_over_gap"),
        (pl.col("last_buy_ago") / (pl.col("buy_gap_mean") + 1.0)).alias("r_gmv_over_gapm"),
        # ускоряется или замедляется покупательский ритм
        (pl.col("buy_gap_last3") - pl.col("buy_gap_mean")).alias("buy_gap_trend"),
        # свежие покупки крупнее или мельче собственного среднего
        (pl.col("buy_gmv_last3") - pl.col("buy_gmv_mean")).alias("buy_gmv_trend"),
        (pl.col("n_buy_days") / (pl.col("first_buy_ago") + 1.0)).alias("buy_rate_since_first"),
    ).sort("user_id")


def phantom_recency(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    span: int = 365,
) -> pl.DataFrame:
    """Recency по СТРОГОМУ определению активности плюс разница с обычным.

    15.2 % строк имеют search = 0 и cat = 0: активность в базе есть, но канал
    не проставлен. Такие дни несут 0.54 % всего GMV, то есть не мусор, но и
    не полноценный визит. Считать ли их активным днём — выбор, который меняет
    recency у заметной доли пользователей.

    Вместо того чтобы выбирать, подаём обе версии и саму разницу между ними:
    величина расхождения информативна сама по себе — она показывает, насколько
    активность пользователя состоит из неатрибутированных дней.
    """
    real = (pl.col("search") == 1) | (pl.col("cat") == 1)
    h = df.filter(pl.col("user_id").is_in(users)
                  & pl.col("d").is_between(anchor - span + 1, anchor))
    f = h.group_by("user_id").agg(
        r_act_strict=(anchor - pl.col("d").filter(real).max()),
        n_real_days=real.sum(),
        n_phantom_days=(~real).sum(),
        r_act_any=(anchor - pl.col("d").max()),
    )
    f = (pl.DataFrame({"user_id": users}).join(f, on="user_id", how="left")
           .with_columns(pl.col("r_act_strict").fill_null(RECENCY_NEVER),
                         pl.col("r_act_any").fill_null(RECENCY_NEVER),
                         pl.col("n_real_days").fill_null(0),
                         pl.col("n_phantom_days").fill_null(0)))
    return f.with_columns(
        # сколько дней «съедает» строгое определение — ноль означает, что
        # последний визит был полноценным
        (pl.col("r_act_strict") - pl.col("r_act_any")).alias("r_act_gap"),
        (pl.col("n_phantom_days")
         / (pl.col("n_real_days") + pl.col("n_phantom_days") + EPS)).alias("phantom_share"),
    ).drop("r_act_any").sort("user_id")
