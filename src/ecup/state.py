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
) -> pl.DataFrame:
    """Полный блок признаков состояния."""
    out = pl.DataFrame({"user_id": users})
    if with_ewma:
        out = out.join(ewma_features(df, anchor, users, span), on="user_id", how="left")
    if with_bayes:
        out = out.join(bayes_filters(df, anchor, users, span), on="user_id", how="left")
    return out.sort("user_id")
