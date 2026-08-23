"""Персональный baseline как OFFSET таргета, а не как колонка дерева.

Сейчас positive-head учится на z - l_plus(A), где l_plus — ОДНА
константа на якорь. Сильнейшая координата пользователя, его GMV за
предыдущие 30 дней, лежит внутри X, и дерево вынуждено кусочно-
постоянными листьями заново строить почти гладкую зависимость
log(1+Y_future) ~ beta·log(1+Y_prev30).

Здесь она выносится в функциональную форму:

    r = z - beta·b - mu_A,   прогноз m = beta·b + mu_last + f(X).

beta оценивается ТОЛЬКО по обучающим якорям, с центрированием внутри
якоря — иначе календарный уровень попадёт в beta.

Отдельный довод в пользу этой конструкции против факторизации
головы: beta·b детерминирован и при усреднении шести моделей не
исчезает, усредняются только остатки. Схлопывания 3e-4 -> 5e-5,
которое случилось с факторизацией, здесь ожидать меньше оснований.

  A  контроль   z - l_plus(A)                      600
  B  один член  b = log1p(gmv_30)                  600
  C  линейный skip из пяти членов, Ridge           600
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
import lightgbm as lgb

ANCH = (348, 378); O = Path('artifacts/neural')
SEEDS = (42, 7, 2026); BUDGET = 600
SKIP = ('gmv_7', 'gmv_30', 'gmv_90', 'ord_30', 'ord_days_90')
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)


def reg(X, y, w, n, s):
    p = dict(ModelConfig(seed=s).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(random_state=s, **p).fit(X, y, sample_weight=w)


def fit_skip(B, z, w, a, lam=1e-6):
    """Веса гладкой части: центрирование ВНУТРИ якоря, взвешенно."""
    Bc = B.copy(); zc = z.copy()
    for k in np.unique(a):
        m = a == k; ww = w[m]
        Bc[m] -= (B[m] * ww[:, None]).sum(0) / ww.sum()
        zc[m] -= (z[m] * ww).sum() / ww.sum()
    G = (Bc * w[:, None]).T @ Bc + lam * np.eye(B.shape[1])
    return np.linalg.solve(G, (Bc * w[:, None]).T @ zc)


for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    B_all = np.log1p(np.column_stack([Xd[c].to_numpy().astype('float64') for c in SKIP]))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = y > 0
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Bv = np.log1p(np.column_stack([val.X[c].to_numpy().astype('float64') for c in SKIP]))
    Xp, wp, ap = X[pos], w[pos], aid[pos]
    zp_raw = np.log1p(y[pos]); zp = zp_raw - zo[pos]; la = max(an)
    Bp = B_all[pos]
    print(f'\n=== якорь {A} · обучающие {an} · покупателей {pos.mean():.3f} ===', flush=True)

    VAR = {}
    for nm, cols in (('B', [1]), ('C', list(range(len(SKIP))))):
        bet = fit_skip(Bp[:, cols], zp_raw, wp, ap)
        s = Bp[:, cols] @ bet; sv = Bv[:, cols] @ bet
        mu = {k: float(((zp_raw - s)[ap == k] * wp[ap == k]).sum() / wp[ap == k].sum())
              for k in np.unique(ap)}
        VAR[nm] = (zp_raw - s - np.array([mu[k] for k in ap]), sv + mu[la], bet)
        print(f'  {nm}: beta {np.array2string(bet, precision=4)} · '
              f'std остатка {VAR[nm][0].std():.4f} (у контроля {zp.std():.4f})', flush=True)

    R = {}
    for s_ in SEEDS:
        t0 = time.perf_counter()
        R[f'A_{s_}'] = reg(Xp, zp, wp, BUDGET, s_).predict(Xva) + last.l_plus
        for nm in ('B', 'C'):
            tgt, off, _ = VAR[nm]
            R[f'{nm}_{s_}'] = reg(Xp, tgt, wp, BUDGET, s_).predict(Xva) + off
        print(f'  сид {s_} за {time.perf_counter()-t0:.0f}с', flush=True)

    o = np.load(O / f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), **R})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0']}).join(t, on='user_id', how='inner')
    p0 = b['p0'].to_numpy(); zz = b['z'].to_numpy()
    sh = lambda k: float((zz - p0 * np.clip(b[k].to_numpy(), 0, None)).std())
    base = np.array([sh(f'A_{s_}') for s_ in SEEDS])
    print(f'  A контроль   ' + ' '.join(f'{x:.5f}' for x in base), flush=True)
    for nm, lab in (('B', 'offset gmv30'), ('C', 'линейный skip')):
        f = np.array([sh(f'{nm}_{s_}') for s_ in SEEDS]); d = base - f
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  {nm} {lab:<14}' + ' '.join(f'{x:.5f}' for x in f) +
              f' · Δ {d.mean():+.5f} · t {d.mean()/max(se,1e-12):+.2f}', flush=True)
    # усреднённые по сидам — проверка на разбавление
    av = lambda k: np.mean([np.clip(b[f'{k}_{s_}'].to_numpy(), 0, None) for s_ in SEEDS], 0)
    sa = float((zz - p0 * av('A')).std())
    print(f'  усреднение 3 сидов: A {sa:.5f} · '
          f'B {sa - float((zz - p0*av("B")).std()):+.5f} · '
          f'C {sa - float((zz - p0*av("C")).std()):+.5f}', flush=True)
    del X, Xva, Xp; gc.collect()
print('\nготово', flush=True)
