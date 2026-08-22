"""Простые представления данных, не покрытые 183 признаками.

183 создаёт впечатление полноты, но это в основном много вариантов
одних и тех же типов статистик: оконные суммы, recency, EWMA,
отношения. Здесь четыре блока, каждый из которых — банальность,
которую мы не построили.

  2. Профиль по дням недели, СВЁРНУТЫЙ ПО СОСТАВУ БУДУЩЕГО ОКНА.
     Это не anchor_doy: модель узнаёт не «сейчас январь», а
     «этот пользователь покупает по субботам, а впереди пять суббот».
  3. Возрасты последних k событий. recency=2 при возрастах 2,6,9 и
     при 2,25,70 — совершенно разные состояния, а recency одна.
  4. Тип дня по каналам вместо сумм по каналам. search и cat в
     панели БИНАРНЫЕ, поэтому состояний ровно четыре, включая день
     активности без обоих флагов (15 % строк).
  5. Разброс и серии. 30 поисков как 1+1+...+1 и как 30+0+...+0 —
     разные пользователи, а сумма одна.
"""
from __future__ import annotations
import numpy as np, polars as pl

HORIZON = 30


def simple_features(df: pl.DataFrame, anchor: int,
                    max_history: int = 300) -> pl.DataFrame:
    # день A ВКЛЮЧЁН: канон в features.py — is_between(A-max+1, A),
    # а цель начинается с A+1. Исключение дня A теряло самый свежий
    # день ровно у возрастов и серий, ради которых блок и строится.
    lo = anchor - max_history + 1
    w = df.filter(pl.col('d').is_between(lo, anchor))
    uid = w.select('user_id').unique()

    # ---------- 2. профиль по дням недели, выровненный на цель ----------
    days = np.arange(lo, anchor + 1)
    hist_cnt = np.bincount(days % 7, minlength=7).astype('float64')
    tgt = np.arange(anchor + 1, anchor + HORIZON + 1)   # цель — (A, A+30]
    tgt_cnt = np.bincount(tgt % 7, minlength=7).astype('float64')
    per = (w.group_by('user_id', 'dow').agg(
               nb=(pl.col('gmv') > 0).sum().cast(pl.Float64),
               no=pl.col('to_ord').sum().cast(pl.Float64),
               ns=pl.col('searches').sum().cast(pl.Float64),
               sg=pl.col('gmv').sum().cast(pl.Float64)))
    hc = pl.DataFrame({'dow': np.arange(7, dtype='int8'),
                       '_hc': hist_cnt, '_tc': tgt_cnt})
    per = per.join(hc, on='dow', how='left')
    # q — частота на КАЛЕНДАРНЫЙ день такого dow, сглаженная
    per = per.with_columns([(pl.col(c) + a) / (pl.col('_hc') + b)
                            for c, a, b in (('nb', 1.0, 7.0), ('no', 1.0, 7.0),
                                            ('ns', 1.0, 7.0), ('sg', 1.0, 7.0))])
    g2 = per.group_by('user_id').agg(
        dw_buy=(pl.col('_tc') * pl.col('nb')).sum(),
        dw_ord=(pl.col('_tc') * pl.col('no')).sum(),
        dw_srch=(pl.col('_tc') * pl.col('ns')).sum(),
        dw_gmv=(pl.col('_tc') * pl.col('sg')).sum(),
        _mb=pl.col('nb').mean(), _sb=pl.col('nb').std(),
        _mg=pl.col('sg').mean())
    g2 = (g2.with_columns(
              # отношение к равномерному ожиданию — чистая доля выравнивания,
              # без общего уровня пользователя
              dw_buy_rel=pl.col('dw_buy') / (HORIZON * pl.col('_mb') + 1e-9),
              dw_gmv_rel=pl.col('dw_gmv') / (HORIZON * pl.col('_mg') + 1e-9),
              dw_conc=pl.col('_sb') / (pl.col('_mb') + 1e-9))
            .drop('_mb', '_sb', '_mg'))

    # ---------- 3. возрасты последних событий ----------
    def ages(flt, pre, ks=(1, 2, 3, 5)):
        s = (w.filter(flt).select('user_id', 'd').unique()
              .sort('user_id', 'd', descending=[False, True])
              .with_columns(j=pl.int_range(pl.len()).over('user_id') + 1))
        out = uid
        for k in ks:
            t = (s.filter(pl.col('j') == k)
                  .select('user_id', **{f'{pre}{k}': (pl.lit(anchor) - pl.col('d'))
                                        .cast(pl.Float64)}))
            out = out.join(t, on='user_id', how='left')
        return out

    g3 = (ages(pl.col('gmv') > 0, 'ba')
          .join(ages(pl.col('to_cart') > 0, 'ca', (1, 2, 3)), on='user_id', how='left')
          .join(ages(pl.lit(True), 'aa', (1, 2)), on='user_id', how='left')
          .join(ages(pl.col('has_search_to_ord') > 0, 'sa', (1,)), on='user_id', how='left')
          .join(ages(pl.col('has_cat_to_ord') > 0, 'ka', (1,)), on='user_id', how='left'))
    g3 = g3.with_columns(ba_d21=pl.col('ba2') - pl.col('ba1'),
                         ba_d32=pl.col('ba3') - pl.col('ba2'),
                         ba_d53=pl.col('ba5') - pl.col('ba3'),
                         ca_d21=pl.col('ca2') - pl.col('ca1'),
                         aa_d21=pl.col('aa2') - pl.col('aa1'))

    # ---------- 4. тип дня по каналам ----------
    st = w.with_columns(
        _s=(pl.col('search') > 0), _c=(pl.col('cat') > 0),
        _o=(pl.col('to_ord') > 0))
    st = st.with_columns(state=pl.when(pl.col('_s') & pl.col('_c')).then(3)
                                 .when(pl.col('_s')).then(1)
                                 .when(pl.col('_c')).then(2).otherwise(0))
    g4 = uid
    for win in (30, 90):
        t = (st.filter(pl.col('d') >= anchor - win).group_by('user_id').agg(
                 **{f'ds_so_{win}': (pl.col('state') == 1).sum().cast(pl.Float64),
                    f'ds_co_{win}': (pl.col('state') == 2).sum().cast(pl.Float64),
                    f'ds_bo_{win}': (pl.col('state') == 3).sum().cast(pl.Float64),
                    f'ds_no_{win}': (pl.col('state') == 0).sum().cast(pl.Float64)}))
        g4 = g4.join(t, on='user_id', how='left')
    # условная вероятность заказа по типу дня, за 180 дней
    c180 = st.filter(pl.col('d') >= anchor - 180)
    for nm, k in (('so', 1), ('co', 2), ('bo', 3)):
        t = (c180.filter(pl.col('state') == k).group_by('user_id')
                 .agg(**{f'ds_p_{nm}': (pl.col('_o').sum().cast(pl.Float64) + 0.5) /
                                       (pl.len() + 3.0)}))
        g4 = g4.join(t, on='user_id', how='left')
    # переходы между соседними АКТИВНЫМИ днями
    tr = (c180.sort('user_id', 'd')
              .with_columns(ps=pl.col('_s').shift(1).over('user_id'),
                            pc=pl.col('_c').shift(1).over('user_id'))
              .drop_nulls(['ps', 'pc']))
    t = tr.group_by('user_id').agg(
        _n=pl.len().cast(pl.Float64),
        _ss=(pl.col('ps') & pl.col('_s')).sum().cast(pl.Float64),
        _sc=(pl.col('ps') & pl.col('_c')).sum().cast(pl.Float64),
        _cs=(pl.col('pc') & pl.col('_s')).sum().cast(pl.Float64),
        _cc=(pl.col('pc') & pl.col('_c')).sum().cast(pl.Float64))
    t = (t.with_columns([(pl.col(c) / (pl.col('_n') + 1.0)).alias(f'ds_tr{c}')
                         for c in ('_ss', '_sc', '_cs', '_cc')])
          .with_columns(ds_switch=(pl.col('_sc') + pl.col('_cs')) / (pl.col('_n') + 1.0))
          .drop('_n', '_ss', '_sc', '_cs', '_cc'))
    g4 = g4.join(t, on='user_id', how='left')

    # ---------- 5. разброс и серии ----------
    w90 = w.filter(pl.col('d') >= anchor - 90)
    aggs = []
    for c, p in (('searches', 'dv_s'), ('to_cart', 'dv_c'),
                 ('to_ord', 'dv_o'), ('gmv', 'dv_g')):
        e = pl.col(c).cast(pl.Float64)
        aggs += [e.std().alias(f'{p}_std'), e.max().alias(f'{p}_max'),
                 e.quantile(0.9).alias(f'{p}_p90')]
    g5 = w90.group_by('user_id').agg(aggs)
    # серии подряд идущих активных дней: разрыв там, где d - d_prev > 1
    run = (w90.sort('user_id', 'd')
              .with_columns(_gap=(pl.col('d') - pl.col('d').shift(1).over('user_id')))
              .with_columns(_new=(pl.col('_gap').is_null() | (pl.col('_gap') > 1))
                            .cast(pl.Int32))
              .with_columns(_run=pl.col('_new').cum_sum().over('user_id')))
    rl = run.group_by('user_id', '_run').agg(_len=pl.len().cast(pl.Float64),
                                             _end=pl.col('d').max())
    g5 = g5.join(rl.group_by('user_id').agg(
        dv_run_max=pl.col('_len').max(), dv_run_n=pl.len().cast(pl.Float64),
        dv_run_mean=pl.col('_len').mean(),
        dv_run_cur=pl.col('_len').filter(
            pl.col('_end') == pl.col('_end').max()).first()), on='user_id', how='left')

    out = (uid.join(g2, on='user_id', how='left')
              .join(g3, on='user_id', how='left')
              .join(g4, on='user_id', how='left')
              .join(g5, on='user_id', how='left'))
    return out
