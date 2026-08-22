"""Чувствительность пользователя к общеплатформенным шокам.

Не `market-state`: глобальное состояние одинаково для всех пользователей
и внутри якоря дерево его выучить не может. Новым здесь является
ВЗАИМОДЕЙСТВИЕ «тип пользователя × глобальный шок» — насколько именно
этот пользователь исторически реагирует на всплески спроса площадки.

    g_t   = standardize(log(1+G_t) - trend_t - DOW_t),
    beta  = sum_t w_t g_t x_{u,t} / (lam + sum_t w_t g_t^2).

Тренд ПРИЧИННЫЙ — скользящее среднее назад. Центрированное окно
затянуло бы в g_t для t <= A информацию о днях после A, а якоря
288/318/348 сдают экзамен именно на будущем.
"""
from __future__ import annotations
import numpy as np, polars as pl

# затухание через exp, а не 0.5 ** expr: во втором случае Python зовёт
# float.__pow__ с выражением polars, и это уходит в UDF без объявленного
# типа возврата, который polars 1.43 отказывается выполнять

QCOLS = ('searches', 'to_cart', 'to_ord', 'gmv')
HALFLIVES = (60.0, 180.0)


def global_shock(df: pl.DataFrame, upto: int, win: int = 28) -> pl.DataFrame:
    """Дневной шок спроса по всем пользователям, только по дням < upto."""
    g = (df.filter(pl.col('d') < upto)
           .group_by('d').agg(pl.col('gmv').sum().alias('G'),
                              pl.col('dow').first().alias('dow'))
           .sort('d')
           .with_columns(lg=pl.col('G').log1p()))
    # причинный тренд: среднее по win предыдущим дням, включая текущий
    g = g.with_columns(trend=pl.col('lg').rolling_mean(win, min_periods=7))
    g = g.with_columns(r=pl.col('lg') - pl.col('trend')).drop_nulls('r')
    dow = g.group_by('dow').agg(pl.col('r').mean().alias('dm'))
    g = g.join(dow, on='dow').with_columns(u=pl.col('r') - pl.col('dm'))
    s = g['u'].std()
    return g.select('d', gt=(pl.col('u') - g['u'].mean()) / (s if s > 0 else 1.0))


def shock_betas(df: pl.DataFrame, anchor: int, max_history: int = 300,
                lam: float = 1.0) -> pl.DataFrame:
    """По одному beta на пару (величина, период полураспада)."""
    g = global_shock(df, anchor)
    lo = anchor - max_history
    s = (df.filter((pl.col('d') >= lo) & (pl.col('d') < anchor))
           .join(g.filter(pl.col('d') >= lo), on='d', how='inner'))
    out = None
    for hl in HALFLIVES:
        w = ((pl.col('d') - anchor) * (np.log(2.0) / hl)).exp()
        agg = [(w * pl.col('gt') * pl.col(c)).sum().alias(f'shk_{c}_{int(hl)}')
               for c in QCOLS]
        # знаменатель: нормировка на собственную энергию пользователя,
        # иначе beta мерит размер пользователя, а не его отзывчивость
        agg += [((w * pl.col(c)).sum() + lam).alias(f'_nrm_{c}_{int(hl)}')
                for c in QCOLS]
        r = s.group_by('user_id').agg(agg)
        r = r.with_columns([(pl.col(f'shk_{c}_{int(hl)}') /
                             pl.col(f'_nrm_{c}_{int(hl)}')).alias(f'shk_{c}_{int(hl)}')
                            for c in QCOLS]).drop([f'_nrm_{c}_{int(hl)}' for c in QCOLS])
        out = r if out is None else out.join(r, on='user_id', how='left')
    return out
