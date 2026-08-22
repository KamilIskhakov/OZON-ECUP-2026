"""Факторизация положительного GMV: Y = N * (Y/N) против одной регрессии.

На покупателях  log Y = log N + log(Y/N)  ТОЧНО, поэтому в бесконечно
гибком пределе факторизация не меняет байесовский прогноз вовсе:

    E[log N | X, Y>0] + E[log(Y/N) | X, Y>0] = E[log Y | X, Y>0].

Весь возможный выигрыш возникает только оттого, что конечному бустингу
проще приблизить две более гладкие функции, чем одну. Значит именно это
и надо доказать, а для этого контроль обязан иметь ТУ ЖЕ ёмкость:
200+200 против 400 и 400+400 против 800. Иначе измеряется число деревьев.

Заменяется только m: склейка идёт с уже сохранённым p0, поэтому эффект
классификатора не примешивается.

Целевое окно — (A, A+30], как в make_target. Смещение на день ломало бы
соответствие с oof['y'], относительно которого считается shape.
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
    return (df.filter(pl.col('d').is_between(anchor + 1, anchor + H))
              .group_by('user_id').agg(
                  n_ord=pl.col('to_ord').sum().cast(pl.Float64),
                  n_buy=(pl.col('gmv') > 0).sum().cast(pl.Float64),
                  gmv=pl.col('gmv').sum().cast(pl.Float64)))


def gbdt(X, y, w, n):
    p = dict(ModelConfig(seed=42).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(**p).fit(X, y, sample_weight=w)


glue = lambda p0, s: p0 * np.log1p(np.exp(np.clip(s, -20, 20)))
R = {}
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid)
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    C = (pl.concat([key.filter(pl.col('_a') == a).join(counts(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row').fill_null(0.0))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = C['gmv'].to_numpy() > 0
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    lY = np.log(C['gmv'].to_numpy()[pos]); Xp, wp = X[pos], w[pos]
    print(f'\n=== якорь {A} · обучающие {an} · покупателей {pos.mean():.3f} ===', flush=True)

    P = {}; t0 = time.perf_counter()
    for nm, num in (('buy', C['n_buy'].to_numpy()), ('ord', C['n_ord'].to_numpy())):
        ln = np.log(np.clip(num[pos], 1.0, None)); la = lY - ln
        for cap in (200, 400):
            if nm == 'ord' and cap == 200: continue          # ord только на 400+400
            P[f'{nm}{cap}'] = (gbdt(Xp, ln, wp, cap).predict(Xva) +
                               gbdt(Xp, la, wp, cap).predict(Xva))
    for cap in (400, 800):
        P[f'ctl{cap}'] = gbdt(Xp, lY, wp, cap).predict(Xva)
    print(f'  восемь моделей за {time.perf_counter()-t0:.0f}с', flush=True)

    o = np.load(O / f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), **P})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0'], 'm0': o['m0']}).join(t, on='user_id', how='inner')
    assert len(b) == len(o['user_id'])
    R[A] = {k: b[k].to_numpy() for k in b.columns if k != 'user_id'}
    g = R[A]; z = g['z']
    print(f'  база p0·m0 {(z - g["p0"]*g["m0"]).std():.5f}', flush=True)
    for k in ('buy200', 'ctl400', 'buy400', 'ord400', 'ctl800'):
        print(f'    {k:<8}{(z - glue(g["p0"], g[k])).std():.5f}', flush=True)
    del X, Xva, Xp; gc.collect()

# --- изолированное направление: факторизация ПРОТИВ контроля той же ёмкости
print(f'\n{"="*70}')
for fac, ctl, nm in (('buy200', 'ctl400', 'buy 200+200 против 400'),
                     ('buy400', 'ctl800', 'buy 400+400 против 800'),
                     ('ord400', 'ctl800', 'ord 400+400 против 800')):
    print(f'\n--- {nm}')
    print(f'{"якорь":>7}{"контроль":>11}{"факториз.":>11}{"разница":>10}'
          f'{"C":>10}{"D":>9}{"alpha":>9}')
    Cs, Ds = {}, {}
    for A in ANCH:
        g = R[A]; z = g['z']
        vc, vf = glue(g['p0'], g[ctl]), glue(g['p0'], g[fac])
        e = z - vc; d = vf - vc; d = d - d.mean()
        Cs[A] = float(((e - e.mean()) * d).mean()); Ds[A] = float((d * d).mean())
        print(f'{A:>7}{e.std():>11.5f}{(z-vf).std():>11.5f}'
              f'{e.std()-(z-vf).std():>+10.5f}{Cs[A]:>+10.5f}{Ds[A]:>9.5f}'
              f'{Cs[A]/Ds[A]:>+9.4f}')
    tr = ANCH[:-1]; lam = sum(Cs[a] for a in tr) / sum(Ds[a] for a in tr)
    g = R[378]; z = g['z']
    vc, vf = glue(g['p0'], g[ctl]), glue(g['p0'], g[fac])
    e = z - vc; d = vf - vc; d = d - d.mean()
    s0, s1 = e.std(), (z - (vc + lam * d)).std()
    print(f'  lam по {list(tr)} = {lam:+.5f} · ПЕРЕНОС {s0:.5f} → {s1:.5f}  '
          f'{s0-s1:+.5f}  (оракул {s0-np.sqrt(max(s0*s0-Cs[378]**2/Ds[378],0)):+.5f})')
np.savez_compressed(O/'freq_decomp.npz',
                    **{f'{k}_{A}': R[A][k] for A in ANCH for k in R[A]})
print('\nготово', flush=True)
