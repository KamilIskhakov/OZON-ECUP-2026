"""Незакрытое намерение воронки — запас, а не очередное отношение.

Существующие признаки описывают counts, ratios, recency и EWMA. Чего в
них нет — состояния «свежих корзин накопилось больше, чем у ЭТОГО
пользователя обычно успевает превратиться в заказы»:

    r_CO = (O_180 + a)/(C_180 + b),   I_cart(h) = r_CO·C_EWMA(h) - O_EWMA(h),
    r_SC = (C_180 + a)/(S_180 + b),   I_srch(h) = r_SC·S_EWMA(h) - C_EWMA(h).

Долгая доля конверсии r берётся по 180 дням и играет роль личной нормы;
короткая EWMA — текущий поток. Положительный I означает избыток свежего
намерения над тем, что уже успело пройти дальше по воронке.
"""
from __future__ import annotations
import polars as pl

HALFLIVES = (3.0, 7.0, 21.0)
A_SMOOTH, B_SMOOTH = 1.0, 5.0


def intent_stock(df: pl.DataFrame, anchor: int, max_history: int = 300) -> pl.DataFrame:
    lo = anchor - max_history
    s = df.filter((pl.col('d') >= lo) & (pl.col('d') < anchor))
    long = (s.filter(pl.col('d') >= anchor - 180)
             .group_by('user_id').agg(S=pl.col('searches').sum(),
                                      C=pl.col('to_cart').sum(),
                                      O=pl.col('to_ord').sum())
             .with_columns(r_CO=(pl.col('O') + A_SMOOTH)/(pl.col('C') + B_SMOOTH),
                           r_SC=(pl.col('C') + A_SMOOTH)/(pl.col('S') + B_SMOOTH))
             .select('user_id', 'r_CO', 'r_SC'))
    aggs = []
    for hl in HALFLIVES:
        w = 0.5 ** ((anchor - pl.col('d')) / hl); h = int(hl)
        aggs += [(w*pl.col('searches')).sum().alias(f'_S{h}'),
                 (w*pl.col('to_cart')).sum().alias(f'_C{h}'),
                 (w*pl.col('to_ord')).sum().alias(f'_O{h}')]
    e = s.group_by('user_id').agg(aggs).join(long, on='user_id', how='left')
    e = e.with_columns(pl.col('r_CO').fill_null(0.2), pl.col('r_SC').fill_null(0.2))
    cols = []
    for hl in HALFLIVES:
        h = int(hl)
        cols += [(pl.col('r_CO')*pl.col(f'_C{h}') - pl.col(f'_O{h}')).alias(f'ist_cart_{h}'),
                 (pl.col('r_SC')*pl.col(f'_S{h}') - pl.col(f'_C{h}')).alias(f'ist_srch_{h}')]
    return (e.with_columns(cols)
             .select(['user_id', 'r_CO', 'r_SC'] +
                     [f'ist_{k}_{int(hl)}' for hl in HALFLIVES for k in ('cart', 'srch')]))
