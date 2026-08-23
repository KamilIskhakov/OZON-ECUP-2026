"""Модель числа будущих ПОКУПОЧНЫХ ДНЕЙ на ВСЕХ пользователях.

Нынешняя propensity-head учится на одном бите C = 1[N_buy > 0]:
пользователь с одним покупочным днём и с десятью — одинаковые
положительные примеры. Прошлый Poisson-эксперимент этого не проверял,
потому что обучался на X[Y > 0], то есть нулевые выбрасывались до
обучения и различение Y=0 против Y>0 улучшить не мог.

Здесь на ВСЕХ пользователях:

    N_u = #{ t in (A, A+30] : GMV_ut > 0 },   N_u in {0..30}
    lam = LGBM_poisson(X, N),   s = log(1 + lam)

Никакого p = 1 - exp(-lam): Poisson нужен только как способ
использовать разметку 0,1,2,... вместо 0,1.

ДЕШЁВАЯ ДИАГНОСТИКА ПЕРВОЙ. Снимаем с s всё, что уже есть в p0:

    s = g0 + g1 * logit(p0) + s_perp

Для малого сдвига логита dp ~ p0 (1 - p0) s_perp, поэтому направление
в z-пространстве d = m0 * p0 * (1 - p0) * s_perp, и считаем обычные
C = E[e d], D = E[d^2], alpha = C/D по каждому якорю отдельно.

ПОЛНАЯ СКЛЕЙКА с ОБЯЗАТЕЛЬНЫМ matched-control, иначе легко снова
открыть уже закрытую ветку аффинной калибровки p:

    (1) p0 m0                                   как есть
    (2) sigma(a + b logit p0) m0                калибратор БЕЗ счётчика
    (3) sigma(a + b logit p0 + c s) m0          со счётчиком

Выигрыш (3) над (2) — цена информации о числе покупочных дней.
Параметры подбираются не по logloss, а прямо под нашу задачу:
min sum [z - m0 sigma(.)]^2.

Протокол prequential: счётчик на якоре A учится только на якорях
a + 30 <= A; (a,b,c) с 288+318 -> 348, с 288+318+348 -> 378.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from scipy.optimize import minimize
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
import lightgbm as lgb

O = Path('artifacts/neural'); AN = (288, 318, 348, 378); SEED = 42; NTREE = 600
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def ndays(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + 30))
              .group_by('user_id')
              .agg(pl.col('d').filter(pl.col('gmv') > 0).n_unique().alias('N')))


S = {}
for A in AN:
    tr = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, tr, sp, None, verbose=False)
    w = anchor_weights(aid)
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    N = (pl.concat([key.filter(pl.col('_a') == a).join(ndays(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row')['N'].fill_null(0).to_numpy().astype('float64'))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    p = dict(ModelConfig(seed=SEED).reg_params)
    p.update(objective='poisson', n_estimators=NTREE, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    t0 = time.perf_counter()
    m = lgb.LGBMRegressor(random_state=SEED, **p).fit(X, N, sample_weight=w)
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    lam = np.clip(m.predict(Xva), 0, None)
    S[A] = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(),
                         's': np.log1p(lam).astype('float64')})
    print(f'якорь {A}: обучающие {tr} · N среднее {(N*w).sum()/w.sum():.3f} '
          f'· доля N=0 {(N < .5).mean():.3f} · lam среднее {lam.mean():.3f} '
          f'· {time.perf_counter()-t0:.0f}с', flush=True)
    del X, Xva; gc.collect()

D = {}
for A in AN:
    o = np.load(O / f'oofpm_a{A}.npz')
    b = (pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                       'p0': o['p0'].astype('float64'), 'm0': o['m0'].astype('float64')})
         .join(S[A], on='user_id', how='inner'))
    D[A] = {k: b[k].to_numpy() for k in ('z', 'p0', 'm0', 's')}

print('\n=== диагностика: ресурс счётчика сверх logit(p0) ===', flush=True)
for A in AN:
    d = D[A]; L = lg(d['p0'])
    G = np.column_stack([np.ones(len(L)), L])
    sp_ = d['s'] - G @ np.linalg.lstsq(G, d['s'], rcond=None)[0]
    dd = d['m0'] * d['p0'] * (1 - d['p0']) * sp_
    e = d['z'] - d['p0'] * d['m0']
    C = float(((e - e.mean()) * (dd - dd.mean())).mean()); Dv = float(((dd - dd.mean())**2).mean())
    d['dir'] = dd; d['e'] = e
    print(f'  {A}: corr(s, logit p0) {np.corrcoef(d["s"], L)[0,1]:+.4f} · '
          f'std(s_perp) {sp_.std():.4f} · C {C:+.3e} · D {Dv:.3e} · '
          f'alpha {C/max(Dv,1e-15):+.4f}', flush=True)

print('\n=== полная склейка, prequential ===', flush=True)
def fit(anchors, use_s):
    zz = np.concatenate([D[a]['z'] for a in anchors])
    LL = np.concatenate([lg(D[a]['p0']) for a in anchors])
    mm = np.concatenate([D[a]['m0'] for a in anchors])
    ss = np.concatenate([D[a]['s'] for a in anchors])
    def f(t):
        u = t[0] + t[1] * LL + (t[2] * ss if use_s else 0.0)
        return float(((zz - mm / (1 + np.exp(-u)))**2).mean())
    x0 = np.array([0.0, 1.0, 0.0])
    return minimize(f, x0, method='Nelder-Mead',
                    options=dict(maxiter=4000, xatol=1e-7, fatol=1e-12)).x

for tr, te in ((( 288, 318), 348), ((288, 318, 348), 378)):
    d = D[te]; L = lg(d['p0'])
    sh = lambda v: float((d['z'] - v).std())
    base = sh(d['p0'] * d['m0'])
    t2 = fit(tr, False); t3 = fit(tr, True)
    v2 = sh(d['m0'] / (1 + np.exp(-(t2[0] + t2[1] * L))))
    v3 = sh(d['m0'] / (1 + np.exp(-(t3[0] + t3[1] * L + t3[2] * d['s']))))
    print(f'  {tr} -> {te}', flush=True)
    print(f'    (1) p0 m0              {base:.5f}', flush=True)
    print(f'    (2) калибратор p0      {v2:.5f} · Δ {base-v2:+.5f} · '
          f'a {t2[0]:+.3f} b {t2[1]:+.3f}', flush=True)
    print(f'    (3) со счётчиком       {v3:.5f} · Δ {base-v3:+.5f} · '
          f'a {t3[0]:+.3f} b {t3[1]:+.3f} c {t3[2]:+.3f}', flush=True)
    print(f'    цена счётчика (3)-(2)  {v2-v3:+.5f}', flush=True)
print('\nготово', flush=True)
