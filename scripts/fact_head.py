"""Замена production-головы регрессии на факторизованную при равном бюджете.

Проверяется НЕ направление для подмешивания, а прямая замена:

    shape(p_old · m_old)  против  shape(p_old · m_fact)

Классификатор не трогается вообще, X, якоря, веса и офсеты те же.
Бюджет деревьев равен: production-голова 600, факторизация 300+300.

Офсеты факторизуются по якорю, как того требует production-семантика:
модели учат остатки u - mu_u(A) и v - mu_v(A), а на целевом якоре
уровень возвращается суммой mu_u + mu_v последнего якоря. Прошлый
эксперимент учил сырой log Y без этого и потому не был переносим.

Поправка Йенсена: f(s) = log(1+e^s) выпукла, f'' = sigma(1-sigma),
поэтому log(1+e^E[s]) занижает E[log(1+e^s)] на 0.5·f''·Var(s|X).
Смещение растёт для мелких покупок и может объяснять, почему
факторизованная модель проигрывает production-голове.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
import lightgbm as lgb

ANCH = (348, 378); H = 30; O = Path('artifacts/neural')
SEEDS = (42, 7, 2026, 13, 99)   # парно: производственная голова и обе пары
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
BUDGET = ModelConfig().reg_params.get('n_estimators', 600)


def counts(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + H)).group_by('user_id')
              .agg(n_ord=pl.col('to_ord').sum().cast(pl.Float64),
                   n_buy=(pl.col('gmv') > 0).sum().cast(pl.Float64)))


def reg(X, y, w, n, seed=42):
    p = dict(ModelConfig(seed=seed).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(**p).fit(X, y, sample_weight=w)


def wmean(v, w):
    return float((v * w).sum() / w.sum())


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
    z = np.log1p(val.y)
    o = np.load(O / f'oofpm_a{A}.npz')
    print(f'\n=== якорь {A} · обучающие {an} · бюджет {BUDGET} ===', flush=True)

    Xp, wp, ap = X[pos], w[pos], aid[pos]
    zp = np.log1p(y[pos]) - zo[pos]; lY = np.log(y[pos]); la = max(an)
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

    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(),
                      **{k: v for k, v in res.items()}})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0']}).join(t, on='user_id', how='inner')
    assert len(b) == len(o['user_id'])
    p0 = b['p0'].to_numpy(); zz = b['z'].to_numpy()
    sh = lambda k: float((zz - p0 * b[k].to_numpy()).std())
    base = np.array([sh(f'production_{s}') for s in SEEDS])
    print(f'  production по сидам ' + ' '.join(f'{v:.5f}' for v in base) +
          f' · std {base.std(ddof=1):.5f}', flush=True)
    for nm in ('buy', 'ord'):
        f = np.array([sh(f'{nm}_{s}') for s in SEEDS]); d = base - f
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  {nm}: ' + ' '.join(f'{v:.5f}' for v in f) +
              f'\n    парные Δ ' + ' '.join(f'{v:+.5f}' for v in d) +
              f' · среднее {d.mean():+.5f} · SE {se:.5f} · t {d.mean()/se:+.2f}',
              flush=True)
    del X, Xva, Xp; gc.collect()
print('\nготово', flush=True)
