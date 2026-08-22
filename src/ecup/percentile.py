"""Поперечные ранги: инвариантность признаков к сдвигу распределения.

Абсолютный gmv_90 на разных якорях означает разное — популяция и
общий уровень площадки между окнами плывут. Ранг внутри якоря

    r_{u,A} = (rank_A(x_{u,A}) - 0.5) / N_A

этим сдвигом не затронут по построению. Механизм отличается от всех
остальных проверенных: он не добавляет информации о пользователе,
а меняет систему отсчёта, в которой дерево эту информацию режет.

Список фиксирован ЗАРАНЕЕ и не подбирается: перебор «ранг против
ранга плюс robust-z», «20 против 40 признаков» вернул бы проклятие
победителя, от которого мы сегодня и пострадали.
"""
from __future__ import annotations
import numpy as np, polars as pl

# 24 количественных признака состояния, у которых поперечный смысл есть:
# объём, заказы, дни покупок, запросы и корзины, давность, средний чек,
# быстрая и медленная шкалы, ритм.
PCT_COLS = (
    'gmv_30', 'gmv_60', 'gmv_90', 'gmv_180',
    'ord_30', 'ord_90', 'ord_180',
    'ord_days_90', 'ord_days_180', 'n_buy_days',
    'srch_90', 'srch_180', 'cart_90', 'cart_180',
    'r_gmv', 'r_ord', 'last_buy_ago',
    'aov_mid', 'buy_gmv_mean',
    'ewm_gmv_fast', 'ewm_gmv_slow',
    'gap_mean', 'buy_gap_mean', 'density_mid',
)


def add_percentiles(X: pl.DataFrame, anchor_ids: np.ndarray) -> pl.DataFrame:
    """Ранги считаются ВНУТРИ каждого якоря — в этом весь смысл.

    Ранжирование по объединению якорей вернуло бы ту самую зависимость
    от общего уровня, ради устранения которой признак и строится.
    """
    cols = [c for c in PCT_COLS if c in X.columns]
    Z = X.with_columns(_aid=pl.Series(anchor_ids),
                       _row=pl.int_range(pl.len(), dtype=pl.UInt32))
    parts = []
    for a in sorted(set(anchor_ids)):
        s = Z.filter(pl.col('_aid') == a)
        n = len(s)
        parts.append(s.with_columns([
            ((pl.col(c).rank('average') - 0.5) / n).alias(f'pct_{c}') for c in cols]))
    return (pl.concat(parts, how='vertical_relaxed')
              .sort('_row').drop('_aid', '_row'))
