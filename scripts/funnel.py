"""Параметры воронки вместо сырых сумм.

Структурный аудит установил (все проверки на полной панели):
  has_* тождественно равны [count > 0] — четыре столбца дубликаты;
  ord <= cart ВНУТРИ ДНЯ без единого исключения — воронка строгая цепь;
  to_cart и to_ord точно раскладываются по каналам;
  search и cat бинарны (max = 1).

Вложенность гарантирует, что условные вероятности определены и лежат
в [0,1]. Значит вместо сырых сумм можно дать дереву параметры процесса:

    интенсивность x конверсия x денежная ценность

Это та же смена представления, что сработала у BG/NBD, и она
принципиально отличается от отвергнутых канальных окон: там были
сырые суммы (48 признаков, -0.00030 на 348), здесь условные
конверсии с байесовским сглаживанием.

Отдельно быстрые и медленные окна: разность p_fast - p_slow различает
«перестал искать» и «ищет столько же, но перестал покупать» — наш
общий байесовский фильтр делает это на уровне активности и конверсии,
но не по отдельным воронкам Search и Catalog.
"""
from __future__ import annotations
import numpy as np
import polars as pl

FAST, SLOW = 30, 180
AB = 2.0      # априор Бета для конверсий
KM = 3.0      # усадка чека


def funnel_features(df: pl.DataFrame, anchor: int, users: np.ndarray
                    ) -> tuple[np.ndarray, list[str]]:
    base = pl.DataFrame({'user_id': users})
    cols: list[np.ndarray] = []
    names: list[str] = []
    acc: dict[tuple[str, int], dict] = {}
    for W in (FAST, SLOW):
        lo = anchor - W + 1
        a = (df.filter(pl.col('d').is_between(lo, anchor)).group_by('user_id').agg(
            srch=pl.col('searches').sum().cast(pl.Float64),
            s2c=pl.col('search_to_cart').sum().cast(pl.Float64),
            s2o=pl.col('search_to_ord').sum().cast(pl.Float64),
            gs=pl.col('gmv_search').sum().cast(pl.Float64),
            cd=pl.col('cat').sum().cast(pl.Float64),
            c2c=pl.col('cat_to_cart').sum().cast(pl.Float64),
            c2o=pl.col('cat_to_ord').sum().cast(pl.Float64),
            gc=pl.col('gmv_cat').sum().cast(pl.Float64),
            sd=pl.col('search').sum().cast(pl.Float64)))
        j = base.join(a, on='user_id', how='left')
        v = {k: j[k].fill_null(0.0).to_numpy() for k in
             ('srch', 's2c', 's2o', 'gs', 'cd', 'c2c', 'c2o', 'gc', 'sd')}
        acc[('v', W)] = v
        gm_s = v['gs'].sum() / max(v['s2o'].sum(), 1.0)
        gm_c = v['gc'].sum() / max(v['c2o'].sum(), 1.0)
        out = (
            # интенсивность: сколько событий канала на день окна
            (f'S_rate_{W}', v['srch'] / W),
            (f'S_days_{W}', v['sd'] / W),
            (f'C_days_{W}', v['cd'] / W),
            # конверсии со сглаживанием, определены благодаря вложенности
            (f'S_p_cart_{W}', (v['s2c'] + AB) / (v['srch'] + 2 * AB)),
            (f'S_p_ord_{W}', (v['s2o'] + AB) / (v['s2c'] + 2 * AB)),
            (f'C_p_ord_{W}', (v['c2o'] + AB) / (v['c2c'] + 2 * AB)),
            (f'C_cart_rate_{W}', (v['c2c'] + AB) / (v['cd'] + 2 * AB)),
            # денежная ценность одного заказа канала
            (f'S_val_{W}', np.log1p((v['gs'] + KM * gm_s) / (v['s2o'] + KM))),
            (f'C_val_{W}', np.log1p((v['gc'] + KM * gm_c) / (v['c2o'] + KM))),
            # доля канала в заказах
            (f'S_share_{W}', (v['s2o'] + AB) / (v['s2o'] + v['c2o'] + 2 * AB)),
        )
        for nm, arr in out:
            cols.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
            names.append(nm)
    # быстро минус медленно: различает «перестал искать» и «перестал покупать»
    f, s = acc[('v', FAST)], acc[('v', SLOW)]
    pf = lambda a, b: (a + AB) / (b + 2 * AB)
    for nm, arr in (
        ('d_S_rate', f['srch'] / FAST - s['srch'] / SLOW),
        ('d_S_p_cart', pf(f['s2c'], f['srch']) - pf(s['s2c'], s['srch'])),
        ('d_S_p_ord', pf(f['s2o'], f['s2c']) - pf(s['s2o'], s['s2c'])),
        ('d_C_p_ord', pf(f['c2o'], f['c2c']) - pf(s['c2o'], s['c2c'])),
        ('d_S_share', pf(f['s2o'], f['s2o'] + f['c2o']) - pf(s['s2o'], s['s2o'] + s['c2o'])),
    ):
        cols.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        names.append(nm)
    return np.column_stack(cols).astype('float32'), names
