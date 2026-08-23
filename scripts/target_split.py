"""Две переparameterизации таргета: по горизонту и по каналу.

  A   контроль          одна голова 600 на log(1+Y)
  H   горизонт          3 x 200 на Y[1:10], Y[11:20], Y[21:30]
  S   канал             2 x 300 на Y_search и Y_cat

В обоих случаях прогнозы суммируются в ИСХОДНОЙ шкале, затем log1p.
Это не тождественная RMSLE-факторизация, поэтому никаких ожиданий
заранее — только парный тест при равном бюджете деревьев.

Механизм для горизонта: связь недавнего поведения с GMV через три дня
и через двадцать семь почти наверняка разная, а сейчас одна голова
обязана выучить обе. Для канала: gmv = gmv_search + gmv_cat выполняется
точно, но каналы очень разные — search присутствует в 80.7 % строк,
cat в 15.6 %, оба сразу лишь в 11.5 %.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
import lightgbm as lgb

ANCH = (348, 378); O = Path('artifacts/neural'); SEEDS = (42, 7, 2026); BUDGET = 600
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
HOR = ((1, 10), (11, 20), (21, 30))


def parts(a):
    """GMV в подокнах горизонта и по каналам за (a, a+30]."""
    w = df.filter(pl.col('d').is_between(a + 1, a + 30))
    agg = [ (pl.col('gmv') * pl.col('d').is_between(a + lo, a + hi)).sum()
            .alias(f'h{lo}') for lo, hi in HOR]
    agg += [pl.col('gmv_search').sum().alias('gs'), pl.col('gmv_cat').sum().alias('gc')]
    return w.group_by('user_id').agg(agg)


def reg(X, y, w, n, s):
    p = dict(ModelConfig(seed=s).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1); p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(random_state=s, **p).fit(X, y, sample_weight=w)


for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    P = (pl.concat([key.filter(pl.col('_a') == a).join(parts(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row').fill_null(0.0))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = y > 0
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Xp, wp = X[pos], w[pos]; zp = np.log1p(y[pos]) - zo[pos]
    cols_h = [f'h{lo}' for lo, _ in HOR]; cols_s = ['gs', 'gc']
    sub = {c: np.log1p(P[c].to_numpy()[pos]) for c in cols_h + cols_s}
    for c in cols_h + cols_s:
        print(f'  {c}: доля нулей среди покупателей {(sub[c] < 1e-9).mean():.3f}', flush=True)
    print(f'=== якорь {A} · обучающие {an} ===', flush=True)

    R = {}
    for s in SEEDS:
        t0 = time.perf_counter()
        R[f'A_{s}'] = reg(Xp, zp, wp, BUDGET, s).predict(Xva) + last.l_plus
        R[f'H_{s}'] = np.log1p(sum(np.clip(np.expm1(
            reg(Xp, sub[c], wp, BUDGET // 3, s).predict(Xva)), 0, None) for c in cols_h))
        R[f'S_{s}'] = np.log1p(sum(np.clip(np.expm1(
            reg(Xp, sub[c], wp, BUDGET // 2, s).predict(Xva)), 0, None) for c in cols_s))
        print(f'  сид {s} за {time.perf_counter()-t0:.0f}с', flush=True)

    o = np.load(O / f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), **R})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0']}).join(t, on='user_id', how='inner')
    p0 = b['p0'].to_numpy(); zz = b['z'].to_numpy()
    sh = lambda k: float((zz - p0 * np.clip(b[k].to_numpy(), 0, None)).std())
    base = np.array([sh(f'A_{s}') for s in SEEDS])
    print(f'  A контроль      ' + ' '.join(f'{x:.5f}' for x in base), flush=True)
    for nm, lab in (('H', 'горизонт 3x200'), ('S', 'канал 2x300')):
        f = np.array([sh(f'{nm}_{s}') for s in SEEDS]); d = base - f
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  {nm} {lab:<15}' + ' '.join(f'{x:.5f}' for x in f) +
              f' · Δ {d.mean():+.5f} · t {d.mean()/max(se,1e-12):+.2f}', flush=True)
    del X, Xva, Xp; gc.collect()
print('\nготово', flush=True)
