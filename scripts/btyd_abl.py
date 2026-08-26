"""Три разреза BTYD-v2: что именно сломало якорь 348.

v2 менял ТРИ вещи сразу — размерность чека, уровни на нескольких
горизонтах и форму hazard, — поэтому его смешанный результат
(-0.00009 на 348, +0.00031 на 378) не говорит, какая часть виновата.

Причём исправление AOV — не гипотеза, а исправление размерности:
если частота измеряется числом ЗАКАЗОВ, то и чек должен быть
GMV на заказ. Отвергать его по смешанному замеру нельзя.

Четыре варианта, каждый отличается от V1 ровно одним:

  V1        p_alive, E[N30], AOV на покупочный день, их произведение
  AOVfix    то же, но AOV на ЗАКАЗ (размерность согласована)
  absolute  V1 + E[N7], E[N14], E[N60]        (уровни)
  shape     V1 + E[N7]/E[N30], E[N60]/2E[N30], 30/E[N30]  (форма)

Шлюз задан ДО просмотра: +2e-4 на обоих якорях.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from btyd import rft, fit, predict

O = Path('artifacts/neural'); SEEDS = (42, 7); HIST = 300; EPS = 1e-6; K = 3.0
df = load_panel(); OUT = {}


def parts(A, uid, par=None):
    """Все составляющие сразу; варианты собираются из них без пересчёта."""
    x, tx, T = rft(A, uid)
    buy = T > 0
    if par is None:
        idx = np.flatnonzero(buy)
        sub = np.random.default_rng(0).choice(idx, min(60000, len(idx)), replace=False)
        par, _ = fit(x[sub], tx[sub], T[sub])
    pa = np.zeros(len(x)); N = {}
    pa[buy], _ = predict(par, x[buy], tx[buy], T[buy], t=30)
    for h in (7, 14, 30, 60):
        v = np.zeros(len(x)); _, v[buy] = predict(par, x[buy], tx[buy], T[buy], t=h)
        N[h] = v
    def aov_of(col, flt):
        w = (df.filter((pl.col('d') <= A) & flt).group_by('user_id')
               .agg(s=pl.col('gmv').sum(), n=col))
        j = pl.DataFrame({'user_id': uid}).join(w, on='user_id', how='left')
        gs = j['s'].fill_null(0.0).to_numpy(); nn = j['n'].fill_null(0.0).to_numpy()
        return (gs + K * (gs.sum() / max(nn.sum(), 1))) / (nn + K)
    aov_day = aov_of(pl.col('d').n_unique().cast(pl.Float64), pl.col('gmv') > 0)
    aov_ord = aov_of(pl.col('to_ord').sum().cast(pl.Float64), pl.col('to_ord') > 0)
    return dict(pa=pa, N=N, aov_day=aov_day, aov_ord=aov_ord), par


def build(P, which):
    pa, N = P['pa'], P['N']
    a = P['aov_ord'] if which == 'AOVfix' else P['aov_day']
    cols = [pa, N[30], np.log1p(a), np.log1p(N[30] * a)]
    if which == 'absolute':
        cols += [N[7], N[14], N[60]]
    elif which == 'shape':
        cols += [N[7] / (N[30] + EPS), N[60] / (2 * N[30] + EPS), 30.0 / (N[30] + EPS)]
    return np.nan_to_num(np.column_stack(cols), posinf=0, neginf=0).astype('float32')


VAR = ('V1', 'AOVfix', 'absolute', 'shape')
for A in (348, 378):
    sp = SplitConfig(max_history=HIST, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
    t0 = time.perf_counter(); B = {v: None for v in VAR}
    for a in sorted(set(aid)):
        m = aid == a
        P, _ = parts(int(a), uid_tr[m])
        for v in VAR:
            b = build(P, v)
            if B[v] is None: B[v] = np.zeros((len(y), b.shape[1]), 'float32')
            B[v][m] = b
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Pv, _ = parts(A, val.X['user_id'].to_numpy())
    z = np.log1p(val.y)
    print(f'\n=== ЯКОРЬ {A} · {time.perf_counter()-t0:.0f}с ===', flush=True)
    for v in VAR:
        Xt = np.hstack([X, B[v]]); Xv = np.hstack([Xva, build(Pv, v)])
        ff = feats + [f'{v}_{i}' for i in range(B[v].shape[1])]
        vs = []
        for s in SEEDS:
            hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
                Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
            p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
            vs.append(float((z - np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))).std()))
        OUT[(A, v)] = np.array(vs)
        d = OUT[(A, 'V1')].mean() - np.mean(vs)
        print(f'  {v:<9}{" ".join(f"{q:.5f}" for q in vs)} · среднее {np.mean(vs):.5f}'
              f'{"" if v == "V1" else f" · к V1 {d:+.5f} · парно " + " ".join(f"{q:+.5f}" for q in OUT[(A,"V1")]-np.array(vs))}',
              flush=True)
        del Xt, Xv; gc.collect()
    del X, Xva; gc.collect()
print(f'\n{"вариант":<10}{"к V1 на 348":>14}{"к V1 на 378":>14}{"знак":>9}')
for v in VAR[1:]:
    d1 = OUT[(348, 'V1')].mean() - OUT[(348, v)].mean()
    d2 = OUT[(378, 'V1')].mean() - OUT[(378, v)].mean()
    sg = 'оба +' if min(d1, d2) > 0 else ('оба -' if max(d1, d2) < 0 else 'РАЗНЫЙ')
    print(f'{v:<10}{d1:>+14.5f}{d2:>+14.5f}{sg:>9}')
print('\nшлюз +2e-4 на ОБОИХ якорях, задан до просмотра')
