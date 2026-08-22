"""Денежная механика покупочного цикла.

Нынешние 183 признака — интегралы по фиксированным календарным окнам:
сколько потрачено за 30/60/90, когда покупал, как часто. Совместного
распределения P(v_j, g_j | история) в них нет, а именно оно описывает
одно денежное событие: сколько пользователь обычно кладёт в покупку и
как размер зависит от накопленного интервала.

Покупочный день определён однозначно: gmv > 0 и to_ord > 0 совпадают
на всех 4 736 907 строках панели.

Три группы:
  1) размер покупочного события — распределение самих v_j, не gmv/orders;
  2) связь интервал → размер следующей покупки, регуляризованный наклон;
  3) положение внутри СОБСТВЕННОГО цикла — накопленное поведение
     относительно того, сколько этому пользователю обычно требуется.
"""
from __future__ import annotations
import numpy as np, polars as pl

HL = 180.0        # период полураспада веса покупки, в днях
LAM = 2.0         # усадка наклона: при одной-двух покупках beta -> 0
BIG_K = 1.5       # порог крупной покупки в MAD от медианы (группа 4, задел)


def monetary_features(df: pl.DataFrame, anchor: int,
                      max_history: int = 300) -> pl.DataFrame:
    lo = anchor - max_history
    win = df.filter((pl.col('d') >= lo) & (pl.col('d') < anchor))
    buys = (win.filter(pl.col('gmv') > 0)
               .select('user_id', 'd', 'gmv', 'to_ord')
               .sort('user_id', 'd')
               .with_columns(v=pl.col('gmv').log1p(),
                             aov=(pl.col('gmv') / pl.col('to_ord').clip(1)).log1p()))
    buys = buys.with_columns(
        g=(pl.col('d') - pl.col('d').shift(1).over('user_id')).cast(pl.Float64),
        k=pl.int_range(pl.len()).over('user_id'),
        w=((pl.col('d') - anchor) * (np.log(2.0) / HL)).exp())

    # --- группа 1: распределение размера самого покупочного события
    g1 = buys.group_by('user_id').agg(
        mv_n=pl.len(),
        mv_med=pl.col('v').median(),
        mv_mad=(pl.col('v') - pl.col('v').median()).abs().median(),
        mv_mean=pl.col('v').mean(),
        mv_last=pl.col('v').last(),
        mv_last3=pl.col('v').tail(3).mean(),
        mv_last5=pl.col('v').tail(5).mean(),
        mv_ewma=(pl.col('w') * pl.col('v')).sum() / (pl.col('w').sum() + 1e-9),
        mv_max=pl.col('v').max(),
        mv_top_share=pl.col('gmv').max() / (pl.col('gmv').sum() + 1e-9),
        mv_aov_med=pl.col('aov').median(),
        mv_aov_last=pl.col('aov').last(),
        mv_ord_med=pl.col('to_ord').median().cast(pl.Float64),
        # наклон размера по номеру покупки — растёт ли чек со временем
        _sk=pl.col('k').cast(pl.Float64).sum(), _skk=(pl.col('k').cast(pl.Float64)**2).sum(),
        _sv=pl.col('v').sum(), _skv=(pl.col('k').cast(pl.Float64) * pl.col('v')).sum(),
    )
    n = pl.col('mv_n').cast(pl.Float64)
    g1 = (g1.with_columns(
              mv_trend=((pl.col('_skv') - pl.col('_sk') * pl.col('_sv') / n) /
                        (pl.col('_skk') - pl.col('_sk')**2 / n + 1.0)))
            .drop('_sk', '_skk', '_sv', '_skv')
            .with_columns(mv_last_rel=pl.col('mv_last') - pl.col('mv_med'),
                          mv_disp=pl.col('mv_mad') / (pl.col('mv_med').abs() + 1e-6)))

    # --- группа 2: интервал -> размер следующей покупки
    gp = buys.filter(pl.col('g').is_not_null()).with_columns(x=pl.col('g').log1p())
    S = gp.group_by('user_id').agg(
        _w=pl.col('w').sum(), _wx=(pl.col('w') * pl.col('x')).sum(),
        _wy=(pl.col('w') * pl.col('v')).sum(),
        _wxx=(pl.col('w') * pl.col('x')**2).sum(),
        _wxy=(pl.col('w') * pl.col('x') * pl.col('v')).sum(),
        gv_n=pl.len(), gv_gap_med=pl.col('g').median())
    S = S.with_columns(_mx=pl.col('_wx') / (pl.col('_w') + 1e-9),
                       _my=pl.col('_wy') / (pl.col('_w') + 1e-9))
    S = S.with_columns(
        _sxx=pl.col('_wxx') - pl.col('_w') * pl.col('_mx')**2,
        _sxy=pl.col('_wxy') - pl.col('_w') * pl.col('_mx') * pl.col('_my'))
    g2 = (S.with_columns(gv_beta=pl.col('_sxy') / (LAM + pl.col('_sxx')),
                         gv_cov=pl.col('_sxy') / (pl.col('_w') + 1e-9),
                         gv_xbar=pl.col('_mx'), gv_ybar=pl.col('_my'))
           .select('user_id', 'gv_n', 'gv_gap_med', 'gv_beta', 'gv_cov',
                   'gv_xbar', 'gv_ybar'))

    # --- группа 3: положение внутри собственного цикла
    # cyc = число покупок СТРОГО до этого дня; цикл 0 обрезан краем окна
    cyc = (win.sort('user_id', 'd')
              .with_columns(_b=(pl.col('gmv') > 0).cast(pl.Int32))
              .with_columns(cyc=pl.col('_b').cum_sum().over('user_id') - pl.col('_b')))
    per = cyc.group_by('user_id', 'cyc').agg(
        c_s=pl.col('searches').sum().cast(pl.Float64),
        c_c=pl.col('to_cart').sum().cast(pl.Float64),
        c_a=pl.len().cast(pl.Float64))
    nb = buys.group_by('user_id').agg(nb=pl.len(), last_buy=pl.col('d').max())
    per = per.join(nb, on='user_id', how='inner')
    done = (per.filter((pl.col('cyc') > 0) & (pl.col('cyc') < pl.col('nb')))
               .group_by('user_id').agg(cy_s_med=pl.col('c_s').median(),
                                        cy_c_med=pl.col('c_c').median(),
                                        cy_a_med=pl.col('c_a').median()))
    cur = (per.filter(pl.col('cyc') == pl.col('nb'))
              .select('user_id', cy_s_now='c_s', cy_c_now='c_c', cy_a_now='c_a'))
    g3 = (nb.select('user_id', 'last_buy')
            .join(done, on='user_id', how='left')
            .join(cur, on='user_id', how='left')
            .with_columns([pl.col(c).fill_null(0.0) for c in
                           ('cy_s_now', 'cy_c_now', 'cy_a_now')])
            .with_columns(cy_r=(anchor - pl.col('last_buy')).cast(pl.Float64)))

    out = (g1.join(g2, on='user_id', how='left')
             .join(g3, on='user_id', how='left'))
    out = out.with_columns(
        # прогноз размера следующей покупки из накопленного интервала
        gv_pred=pl.col('gv_ybar') + pl.col('gv_beta') *
                (pl.col('cy_r').log1p() - pl.col('gv_xbar')),
        cy_q_time=pl.col('cy_r') / (pl.col('gv_gap_med') + 1.0),
        cy_q_srch=pl.col('cy_s_now') / (pl.col('cy_s_med') + 1.0),
        cy_q_cart=pl.col('cy_c_now') / (pl.col('cy_c_med') + 1.0),
        cy_q_act=pl.col('cy_a_now') / (pl.col('cy_a_med') + 1.0))
    out = out.with_columns(
        gv_pred_rel=pl.col('gv_pred') - pl.col('mv_med'),
        # опережает ли накопленное намерение календарный ход цикла
        cy_srch_over_time=pl.col('cy_q_srch') / (pl.col('cy_q_time') + 0.1),
        cy_cart_over_time=pl.col('cy_q_cart') / (pl.col('cy_q_time') + 0.1),
        cy_act_over_time=pl.col('cy_q_act') / (pl.col('cy_q_time') + 0.1))
    return out.drop('last_buy')
