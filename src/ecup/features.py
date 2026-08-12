"""Построение признаков на якорь.

Инвариант, который нельзя нарушать: в признаки якоря `T` попадают только дни
`d <= T`. Всё, что после — таргет, и любое его касание это утечка.
"""
from __future__ import annotations

import polars as pl

from .config import SELECTION_SPAN, Windows, d_to_date

EPS = 1e-6
RECENCY_NEVER = 9_999          # «события не было вовсе» — осмысленнее нуля или null


def _rolling_exprs(anchor: int, windows: Windows) -> list[pl.Expr]:
    out: list[pl.Expr] = []
    for w in windows.rolling:
        m = pl.col("d") > anchor - w
        pos = m & (pl.col("gmv") > 0)
        out += [
            pl.col("gmv").filter(m).sum().alias(f"gmv_{w}"),
            pl.col("to_ord").filter(m).sum().alias(f"ord_{w}"),
            pl.col("to_cart").filter(m).sum().alias(f"cart_{w}"),
            pl.col("searches").filter(m).sum().alias(f"srch_{w}"),
            m.sum().alias(f"act_days_{w}"),
            (m & (pl.col("to_ord") > 0)).sum().alias(f"ord_days_{w}"),
            (m & (pl.col("to_cart") > 0)).sum().alias(f"cart_days_{w}"),
            pl.col("gmv").filter(pos).max().alias(f"gmv_max_{w}"),
            pl.col("gmv").filter(pos).mean().alias(f"gmv_mean_pos_{w}"),
            pl.col("gmv").filter(pos).std().alias(f"gmv_std_pos_{w}"),
        ]
    for lo, hi in windows.lagged:
        m = pl.col("d").is_between(anchor - hi + 1, anchor - lo)
        out += [
            pl.col("gmv").filter(m).sum().alias(f"gmv_lag_{lo}_{hi}"),
            pl.col("to_ord").filter(m).sum().alias(f"ord_lag_{lo}_{hi}"),
            m.sum().alias(f"act_days_lag_{lo}_{hi}"),
        ]
    return out


def _channel_recency_exprs(anchor: int, cw: int) -> list[pl.Expr]:
    """Канальный разрез и recency. `cw` — окно канальных признаков."""
    m = pl.col("d") > anchor - cw
    return [
        pl.col("gmv_search").filter(m).sum().alias("gmv_search_c"),
        pl.col("gmv_cat").filter(m).sum().alias("gmv_cat_c"),
        pl.col("search").filter(m).sum().alias("search_days_c"),
        pl.col("cat").filter(m).sum().alias("cat_days_c"),
        pl.col("search_to_ord").filter(m).sum().alias("s2o_c"),
        pl.col("cat_to_ord").filter(m).sum().alias("c2o_c"),
        pl.col("search_to_cart").filter(m).sum().alias("s2c_c"),
        pl.col("cat_to_cart").filter(m).sum().alias("c2c_c"),
        # «фантомные» дни: активность есть, но ни Поиск, ни Каталог не проставлены
        (m & (pl.col("search") == 0) & (pl.col("cat") == 0)).sum().alias("phantom_c"),
        (m & (pl.col("dow") >= 5)).sum().alias("weekend_days_c"),
        # recency: сколько дней назад было последнее событие каждого типа
        (anchor - pl.col("d").max()).alias("r_act"),
        (anchor - pl.col("d").filter(pl.col("to_ord") > 0).max()).alias("r_ord"),
        (anchor - pl.col("d").filter(pl.col("to_cart") > 0).max()).alias("r_cart"),
        (anchor - pl.col("d").filter(pl.col("searches") > 0).max()).alias("r_srch"),
        (anchor - pl.col("d").filter(pl.col("gmv") > 0).max()).alias("r_gmv"),
        (anchor - pl.col("d").min()).alias("hist_span"),
        pl.len().alias("act_days_hist"),
        pl.col("gmv").sum().alias("gmv_hist"),
        (pl.col("gmv") > 0).sum().alias("gmv_days_hist"),
    ]


def _gap_stats(hist: pl.DataFrame) -> pl.DataFrame:
    """Статистики интервалов между визитами — основа нормированного recency.

    Идея из BTYD: 28 дней молчания у того, кто ходит раз в неделю, и у того,
    кто ходит раз в два месяца, — совершенно разные сигналы. Саму BTYD-модель
    здесь строить смысла нет (правило отбора гарантирует, что все «живы»),
    а вот нормировка recency на собственный ритм ничего не стоит.
    """
    return (
        hist.select(["user_id", "d"])
            .sort(["user_id", "d"])
            .with_columns(gap=(pl.col("d") - pl.col("d").shift(1).over("user_id")))
            .drop_nulls("gap")
            .group_by("user_id")
            .agg(
                gap_mean=pl.col("gap").mean(),
                gap_median=pl.col("gap").median(),
                gap_max=pl.col("gap").max(),
                gap_std=pl.col("gap").std(),
            )
    )


def _derived(f: pl.LazyFrame, windows: Windows, cw: int) -> pl.LazyFrame:
    """Отношения и тренды поверх сырых сумм."""
    roll = windows.rolling
    short = roll[min(2, len(roll) - 1)]           # обычно 30
    mid = roll[min(4, len(roll) - 1)]             # обычно 90
    long = roll[-1]

    exprs = [
        # средний чек. ВНИМАНИЕ (01_eda §7.4): сырая корреляция с таргетом
        # положительная, а коэффициент при контроле активности отрицательный.
        # Признак сильный, но интерпретировать его в одиночку нельзя.
        (pl.col(f"gmv_{mid}") / (pl.col(f"ord_{mid}") + EPS)).alias("aov_mid"),
        (pl.col(f"ord_{mid}") / (pl.col(f"cart_{mid}") + EPS)).alias("cvr_cart2ord"),
        (pl.col(f"cart_{mid}") / (pl.col(f"srch_{mid}") + EPS)).alias("cvr_srch2cart"),
        (pl.col(f"ord_days_{mid}") / (pl.col(f"act_days_{mid}") + EPS)).alias("buy_rate"),
        (pl.col(f"cart_days_{mid}") / (pl.col(f"act_days_{mid}") + EPS)).alias("cart_rate"),
        (pl.col(f"act_days_{mid}") / float(mid)).alias("density_mid"),
        (pl.col(f"act_days_{long}") / float(long)).alias("density_long"),
        (pl.col(f"gmv_{mid}") / (pl.col(f"act_days_{mid}") + EPS)).alias("gmv_per_active"),
        (pl.col(f"srch_{mid}") / (pl.col(f"act_days_{mid}") + EPS)).alias("srch_per_active"),
        (pl.col("gmv_search_c") / (pl.col("gmv_search_c") + pl.col("gmv_cat_c") + EPS))
            .alias("share_search_gmv"),
        (pl.col("search_days_c") / (pl.col("search_days_c") + pl.col("cat_days_c") + EPS))
            .alias("share_search_days"),
        (pl.col("phantom_c") / (pl.col("act_days_hist") + EPS)).alias("phantom_rate"),
        (pl.col("weekend_days_c") / (pl.col("act_days_hist") + EPS)).alias("weekend_rate"),
        # нормированный recency: молчание относительно собственного ритма
        (pl.col("r_act") / (pl.col("gap_median") + 1.0)).alias("r_act_norm"),
        (pl.col("r_act") / (pl.col("gap_mean") + 1.0)).alias("r_act_norm_mean"),
        # доля истории, прошедшая с последней покупки
        (pl.col("r_gmv") / (pl.col("hist_span") + 1.0)).alias("r_gmv_rel"),
    ]
    if f"gmv_lag_{short}_60" in f.collect_schema().names():
        exprs += [
            ((pl.col(f"gmv_{short}") + 1).log()
             - (pl.col(f"gmv_lag_{short}_60") + 1).log()).alias("trend_gmv"),
            ((pl.col(f"act_days_{short}") + 1).log()
             - (pl.col(f"act_days_lag_{short}_60") + 1).log()).alias("trend_act"),
        ]
    if len(roll) >= 5:
        exprs += [
            (pl.col(f"gmv_{short}") / (pl.col(f"gmv_{mid}") + EPS)).alias("gmv_short_over_mid"),
            (pl.col(f"act_days_{short}") / (pl.col(f"act_days_{mid}") + EPS))
                .alias("act_short_over_mid"),
        ]
    return f.with_columns(exprs)


def build_features(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    max_history: int = 365,
    windows: Windows | None = None,
    with_calendar: bool = True,
) -> pl.DataFrame:
    """Признаки популяции `users` на момент `anchor`.

    `max_history` — глубина окна признаков. Окна длиннее неё обрезаются, иначе
    они молча считались бы по неполной истории и дублировали самое длинное
    допустимое окно.
    """
    windows = (windows or Windows()).clipped(max_history)
    cw = min(90, max_history)

    hist = df.filter(
        pl.col("user_id").is_in(users)
        & pl.col("d").is_between(anchor - max_history + 1, anchor)
    )

    agg = hist.group_by("user_id").agg(
        _rolling_exprs(anchor, windows) + _channel_recency_exprs(anchor, cw)
    )
    agg = agg.join(_gap_stats(hist), on="user_id", how="left")

    f = (
        pl.DataFrame({"user_id": users})
          .join(agg, on="user_id", how="left")
          .lazy()
    )
    f = _derived(f, windows, cw)
    out = f.collect()

    recency_cols = [c for c in out.columns if c.startswith("r_") and out[c].dtype.is_integer()]
    out = out.with_columns(
        [pl.col(c).fill_null(RECENCY_NEVER).cast(pl.Int32) for c in recency_cols]
        + [pl.col(c).fill_null(0.0) for c in out.columns
           if c != "user_id" and c not in recency_cols]
    )

    # Сколько истории реально стояло за агрегатами. На ранних якорях окно
    # обрезается началом данных, и модель должна это видеть, иначе «gmv_365»
    # у разных якорей означает разное.
    out = out.with_columns(avail_history=pl.lit(min(max_history, anchor + 1), dtype=pl.Int16))

    if with_calendar:
        dt = d_to_date(anchor)
        out = out.with_columns(
            anchor_d=pl.lit(anchor, dtype=pl.Int32),
            anchor_month=pl.lit(dt.month, dtype=pl.Int8),
            anchor_dow=pl.lit(dt.weekday(), dtype=pl.Int8),
            anchor_doy=pl.lit(dt.timetuple().tm_yday, dtype=pl.Int16),
        )
    return out.sort("user_id")


ANCHOR_CALENDAR = ("anchor_month", "anchor_dow", "anchor_doy")


def feature_names(
    frame: pl.DataFrame,
    drop_anchor_id: bool = True,
    drop_anchor_calendar: bool = True,
) -> list[str]:
    """Колонки-признаки.

    `anchor_d` исключается всегда: это индекс времени, по которому модель
    выучила бы «в свежих якорях таргет такой-то» и не смогла бы
    экстраполировать на боевой якорь, лежащий правее всего трейна.

    Календарь якоря исключается по той же причине, только тоньше. Внутри
    якоря это константы, то есть ещё один его идентификатор. Дни года
    обучающих якорей — 199, 229, ..., 349, 14; боевого — 44. Дерево отнесёт
    44 в бин между 14 и 199 и применит произвольно выбранную
    январско-июльскую интерполяцию. Сезонность задаётся явно через
    p_target и m_offset, а не выучивается по номеру дня.

    `avail_history` остаётся: она объясняет усечённые окна ранних якорей.
    """
    drop = {"user_id", "y"}
    if drop_anchor_id:
        drop.add("anchor_d")
    if drop_anchor_calendar:
        drop.update(ANCHOR_CALENDAR)
    return [c for c in frame.columns if c not in drop]
