"""Факторизация головы регрессии у CatBoost — вторая половина бэкбона.

У LGB замена головы дала +0.00037/+0.00048 на 348 и +0.00031/+0.00026
на 378 при равном бюджете. На ансамбле осталось +0.00007, потому что
вес LGB равен 0.4, а двенадцать CatBoost сохранили обычную голову.
Здесь проверяется вторая половина.

Режим тот же, что валидирован у LGB: фиксированный бюджет без ранней
остановки, 600 против 300+300, классификатор не участвует.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, load_panel, build_anchor, build_training_set,
                  to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.catboost_model import CatBoostConfig

ANCH = (348, 378); H = 30; O = Path('artifacts/neural')
SEEDS = (42, 7, 2026); BUDGET = 600
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)


def counts(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + H)).group_by('user_id')
              .agg(n_ord=pl.col('to_ord').sum().cast(pl.Float64),
                   n_buy=(pl.col('gmv') > 0).sum().cast(pl.Float64)))


def reg(X, y, w, n, seed):
    from catboost import CatBoostRegressor, Pool
    p = dict(CatBoostConfig(seed=seed).reg_params); p['iterations'] = n
    m = CatBoostRegressor(random_seed=seed, **p)
    m.fit(Pool(X, label=y, weight=w), verbose=False)
    return m


wmean = lambda v, w: float((v * w).sum() / w.sum())
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    C = (pl.concat([key.filter(pl.col('_a') == a).join(counts(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row').fill_null(0.0))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = y > 0
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Xp, wp, ap = X[pos], w[pos], aid[pos]
    zp = np.log1p(y[pos]) - zo[pos]; lY = np.log(y[pos]); la = max(an)
    print(f'\n=== якорь {A} · обучающие {an} · бюджет {BUDGET} ===', flush=True)

    res = {}
    for sd in SEEDS:
        t0 = time.perf_counter()
        res[f'production_{sd}'] = reg(Xp, zp, wp, BUDGET, sd).predict(Xva) + last.l_plus
        for nm, num in (('buy', C['n_buy'].to_numpy()[pos]),
                        ('ord', C['n_ord'].to_numpy()[pos])):
            u = np.log(np.clip(num, 1.0, None)); v = lY - u
            mu_u = {a: wmean(u[ap == a], wp[ap == a]) for a in sorted(set(ap))}
            mu_v = {a: wmean(v[ap == a], wp[ap == a]) for a in sorted(set(ap))}
            ou = np.array([mu_u[a] for a in ap]); ov = np.array([mu_v[a] for a in ap])
            s = (reg(Xp, u - ou, wp, BUDGET // 2, sd).predict(Xva) + mu_u[la] +
                 reg(Xp, v - ov, wp, BUDGET // 2, sd).predict(Xva) + mu_v[la])
            res[f'{nm}_{sd}'] = np.log1p(np.exp(np.clip(s, -20, 20)))
        print(f'  сид {sd} за {time.perf_counter()-t0:.0f}с', flush=True)

    o = np.load(O / f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), **res})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0']}).join(t, on='user_id', how='inner')
    p0 = b['p0'].to_numpy(); zz = b['z'].to_numpy()
    sh = lambda k: float((zz - p0 * np.clip(b[k].to_numpy(), 0, None)).std())
    base = np.array([sh(f'production_{s}') for s in SEEDS])
    print(f'  production ' + ' '.join(f'{v:.5f}' for v in base), flush=True)
    for nm in ('buy', 'ord'):
        f = np.array([sh(f'{nm}_{s}') for s in SEEDS]); d = base - f
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  {nm}: ' + ' '.join(f'{v:.5f}' for v in f) +
              f' · парные Δ ' + ' '.join(f'{v:+.5f}' for v in d) +
              f' · среднее {d.mean():+.5f} · t {d.mean()/max(se,1e-12):+.2f}', flush=True)
    del X, Xva, Xp; gc.collect()
print('\nготово', flush=True)
