"""Канальные и воронковые величины на НЕСКОЛЬКИХ окнах.

Систематический пробел в наборе признаков. Ось окон 7/14/30/60/90/180
применена к GMV, заказам, корзинам и поискам и составляет ядро набора.
К каналам и воронке она НЕ применена: gmv_search_c, gmv_cat_c, s2o_c,
c2o_c, s2c_c, c2c_c, phantom_c, weekend_days_c существуют только как
накопленные за одно окно (_channel_recency_exprs берёт один cw).

При этом каналы ведут себя по-разному: поиск присутствует в 80.7 %
строк, каталог в 15.6 %, оба сразу лишь в 11.5 %. Их динамика на
разных горизонтах — наблюдаемая, которую модель сейчас не видит.

Блок: 12 величин на окно x 4 окна = 48 признаков.
"""
from __future__ import annotations
import numpy as np
import polars as pl

WINS = (7, 30, 90, 180)


def channel_windows(df: pl.DataFrame, anchor: int, users: np.ndarray
                    ) -> tuple[np.ndarray, list[str]]:
    base = pl.DataFrame({'user_id': users})
    cols: list[np.ndarray] = []
    names: list[str] = []
    for W in WINS:
        lo = anchor - W + 1
        a = (df.filter(pl.col('d').is_between(lo, anchor)).group_by('user_id').agg(
            gs=pl.col('gmv_search').sum().cast(pl.Float64),
            gc=pl.col('gmv_cat').sum().cast(pl.Float64),
            sd=pl.col('search').sum().cast(pl.Float64),
            cd=pl.col('cat').sum().cast(pl.Float64),
            s2o=pl.col('search_to_ord').sum().cast(pl.Float64),
            c2o=pl.col('cat_to_ord').sum().cast(pl.Float64),
            s2c=pl.col('search_to_cart').sum().cast(pl.Float64),
            c2c=pl.col('cat_to_cart').sum().cast(pl.Float64),
            ph=((pl.col('search') == 0) & (pl.col('cat') == 0)).sum().cast(pl.Float64),
            we=(pl.col('dow') >= 5).sum().cast(pl.Float64),
            sr=pl.col('searches').sum().cast(pl.Float64),
            g=pl.col('gmv').sum().cast(pl.Float64),
            o=pl.col('to_ord').sum().cast(pl.Float64)))
        j = base.join(a, on='user_id', how='left')
        v = {k: j[k].fill_null(0.0).to_numpy() for k in
             ('gs', 'gc', 'sd', 'cd', 's2o', 'c2o', 's2c', 'c2c', 'ph', 'we', 'sr', 'g', 'o')}
        eps = 1e-9
        out = (
            ('gmv_search', np.log1p(v['gs'])),
            ('gmv_cat', np.log1p(v['gc'])),
            ('search_days', v['sd']),
            ('cat_days', v['cd']),
            ('s2o', np.log1p(v['s2o'])),
            ('c2o', np.log1p(v['c2o'])),
            ('s2c', np.log1p(v['s2c'])),
            ('c2c', np.log1p(v['c2c'])),
            ('phantom', v['ph']),
            ('weekend', v['we']),
            # доли и конверсии в том же окне — то, чего нет вовсе
            ('share_search_gmv', v['gs'] / (v['g'] + eps)),
            ('s2o_rate', v['s2o'] / (v['sr'] + eps)),
        )
        for nm, arr in out:
            cols.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
            names.append(f'{nm}_{W}')
    return np.column_stack(cols).astype('float32'), names
