"""Факторизация целевого окна: Y = N * AOV вместо одного m.

Существующий hurdle даёт E[z|x] = p(x)·m(x), где m — единственное
число на всю положительную сумму за 30 дней. Но для покупателя

    log Y = log N + log AOV

выполняется ТОЧНО. Поэтому заменяется только m: две регрессии на
покупателях, склейка с уже сохранённым p0. Так сравнение с p0·m0
изолирует эффект факторизации и не смешивает его с изменением
классификатора.

Две факторизации: по числу заказов и по числу покупочных дней.
Вторая устойчивее — покупочных дней меньше и они менее шумны.

Протокол как у XGBoost: walk-forward 288/318/348/378, коэффициент
смеси оценивается по первым трём якорям и замораживается для 378.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
import lightgbm as lgb

ANCH = (288, 318, 348, 378); H = 30; O = Path('artifacts/neural')
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)


def counts(anchor):
    """N заказов, N покупочных дней и сумма в окне [A, A+30)."""
    return (df.filter((pl.col('d') >= anchor) & (pl.col('d') < anchor + H))
              .group_by('user_id').agg(
                  n_ord=pl.col('to_ord').sum().cast(pl.Float64),
                  n_buy=(pl.col('gmv') > 0).sum().cast(pl.Float64),
                  gmv=pl.col('gmv').sum().cast(pl.Float64)))


def gbdt(X, y, w, seed=42, n=400):
    p = dict(ModelConfig(seed=seed).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(**p).fit(X, y, sample_weight=w)


R = {}
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid)
    # цели считаются для каждого обучающего якоря свои
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    parts = [key.filter(pl.col('_a') == a).join(counts(a), on='user_id', how='left')
             for a in sorted(set(aid))]
    C = pl.concat(parts, how='vertical_relaxed').sort('_row').fill_null(0.0)
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = C['gmv'].to_numpy() > 0
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    print(f'\n=== якорь {A} · обучающие {an} · строк {len(y):,} · '
          f'покупателей {pos.mean():.3f} ===', flush=True)

    t0 = time.perf_counter(); P = {}
    for nm, num in (('ord', C['n_ord'].to_numpy()), ('buy', C['n_buy'].to_numpy())):
        # log N и log AOV на покупателях; сумма даёт log Y точно
        ln = np.log(np.clip(num[pos], 1.0, None))
        la = np.log(C['gmv'].to_numpy()[pos]) - ln
        P[f'n_{nm}'] = gbdt(X[pos], ln, w[pos]).predict(Xva)
        P[f'a_{nm}'] = gbdt(X[pos], la, w[pos]).predict(Xva)
    # контроль: одна регрессия на log Y, та же выборка и та же ёмкость
    P['m_ctl'] = gbdt(X[pos], np.log(C['gmv'].to_numpy()[pos]), w[pos]).predict(Xva)
    print(f'  пять моделей за {time.perf_counter()-t0:.0f}с', flush=True)

    o = np.load(O / f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(),
                      **{k: v for k, v in P.items()}})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0'], 'm0': o['m0']}).join(t, on='user_id', how='inner')
    assert len(b) == len(o['user_id'])
    g = {k: b[k].to_numpy() for k in b.columns if k != 'user_id'}
    R[A] = g
    base = g['p0'] * g['m0']; z = g['z']
    print(f'  база p0·m0 {(z-base).std():.5f}', flush=True)
    for nm in ('ord', 'buy'):
        v = g['p0'] * np.log1p(np.exp(np.clip(g[f'n_{nm}'] + g[f'a_{nm}'], -20, 20)))
        print(f'  факторизация {nm}: {(z-v).std():.5f}', flush=True)
    vc = g['p0'] * np.log1p(np.exp(np.clip(g['m_ctl'], -20, 20)))
    print(f'  контроль (одна регрессия): {(z-vc).std():.5f}', flush=True)
    del X, Xva; gc.collect()

print(f'\n{"="*64}')
for nm in ('ord', 'buy', 'ctl'):
    C_, D_ = {}, {}
    print(f'\n--- {nm}')
    print(f'{"якорь":>7}{"база":>10}{"своё":>10}{"C":>10}{"D":>9}{"alpha":>9}')
    for A in ANCH:
        g = R[A]; base = g['p0'] * g['m0']; z = g['z']
        s = g['m_ctl'] if nm == 'ctl' else g[f'n_{nm}'] + g[f'a_{nm}']
        v = g['p0'] * np.log1p(np.exp(np.clip(s, -20, 20)))
        e = z - base; d = v - base; d = d - d.mean(); ec = e - e.mean()
        C_[A], D_[A] = float((ec*d).mean()), float((d*d).mean())
        print(f'{A:>7}{e.std():>10.5f}{(z-v).std():>10.5f}'
              f'{C_[A]:>+10.5f}{D_[A]:>9.5f}{C_[A]/D_[A]:>+9.4f}')
    tr = ANCH[:-1]; lam = sum(C_[a] for a in tr) / sum(D_[a] for a in tr)
    g = R[378]; base = g['p0']*g['m0']; z = g['z']
    s = g['m_ctl'] if nm == 'ctl' else g[f'n_{nm}'] + g[f'a_{nm}']
    v = g['p0'] * np.log1p(np.exp(np.clip(s, -20, 20)))
    d = v - base; d = d - d.mean(); e = z - base
    s0, s1 = e.std(), (z - (base + lam*d)).std()
    print(f'  lam по {list(tr)} = {lam:+.5f} · ПЕРЕНОС {s0:.5f} → {s1:.5f}  {s0-s1:+.5f}'
          f'  (оракул {s0-np.sqrt(max(s0*s0-C_[378]**2/D_[378],0)):+.5f})')
np.savez_compressed(O/'freq_decomp.npz',
                    **{f'{k}_{A}': R[A][k] for A in ANCH for k in R[A]})
print('\nготово', flush=True)
