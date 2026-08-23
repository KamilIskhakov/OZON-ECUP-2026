"""Вентиль факторизации по ПРЕДСКАЗАННОМУ среднему чеку.

Прошлый вентиль строился по предсказанному уровню m и не дал ничего
(+0.00001), потому что вредная страта определялась истинным z. Теперь
известно, что структура выигрыша монотонна по AOV = Y / N, а не по N
и не по уровню, и есть приличная оценка N (R^2 = 0.40 по log(1+N)
среди покупателей). Значит доступна величина

    AOV_hat = m_ctl - log(1 + lam_hat)

вычислимая в момент прогноза. Счётчик обучается ТОЛЬКО НА
ПОКУПАТЕЛЯХ, prequential.

Сначала печатается corr(AOV_hat, AOV) среди покупателей: если она
низкая, вентиль заведомо не выделит верхний квинтиль и ветка
закрывается до подбора весов.

Вентиль обучается на 288+318 по квантильным корзинам AOV_hat и
применяется к 348 и 378 против ЕДИНОГО lam на тех же якорях.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
import lightgbm as lgb

O = Path('artifacts/neural'); AN = (288, 318, 348, 378); SEED = 42; NB = 8
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
F = np.load(O / 'freq_decomp.npz')


def ndays(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + 30))
              .group_by('user_id')
              .agg(pl.col('d').filter(pl.col('gmv') > 0).n_unique().alias('N')))


D = {}
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
    pos = N > 0.5
    p = dict(ModelConfig(seed=SEED).reg_params)
    p.update(objective='poisson', n_estimators=600, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    t0 = time.perf_counter()
    m = lgb.LGBMRegressor(random_state=SEED, **p).fit(X[pos], N[pos], sample_weight=w[pos])
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    lam = np.clip(m.predict(Xva), 1e-6, None)
    nv = (pl.DataFrame({'user_id': val.X['user_id'].to_numpy()})
            .join(ndays(A), on='user_id', how='left')['N'].fill_null(0).to_numpy().astype('float64'))
    z = F[f'z_{A}']; p0 = F[f'p0_{A}']
    g = lambda k: np.log1p(np.exp(np.clip(F[f'{k}_{A}'], -20, 20)))
    mc, mf = g('ctl400'), g('buy200')
    ahat = mc - np.log1p(lam)
    b = nv > 0.5
    atrue = np.full(len(nv), np.nan); atrue[b] = np.log(np.expm1(z[b]) / nv[b] + 1e-9)
    c = np.corrcoef(ahat[b], atrue[b])[0, 1]
    print(f'якорь {A}: corr(AOV_hat, AOV) среди покупателей {c:+.4f} · '
          f'std AOV_hat {ahat.std():.3f} · std AOV {atrue[b].std():.3f} · '
          f'{time.perf_counter()-t0:.0f}с', flush=True)
    D[A] = dict(z=z, p0=p0, mc=mc, mf=mf, a=ahat)
    del X, Xva; gc.collect()

U = np.concatenate([D[a]['a'] for a in (288, 318)])
q = np.quantile(U, np.linspace(0, 1, NB + 1))[1:-1]
num = np.zeros(NB); den = np.zeros(NB)
for a in (288, 318):
    d = D[a]; e = d['z'] - d['p0'] * d['mc']; dd = d['p0'] * (d['mf'] - d['mc'])
    k = np.clip(np.digitize(d['a'], q), 0, NB - 1)
    for j in range(NB):
        m_ = k == j
        num[j] += float(((e[m_] - e[m_].mean()) * (dd[m_] - dd[m_].mean())).sum())
        den[j] += float(((dd[m_] - dd[m_].mean())**2).sum())
lam_b = num / np.maximum(den, 1e-12); flat = num.sum() / den.sum()
print(f'\nвентиль по квантилям AOV_hat (обучен на 288+318):', flush=True)
for j in range(NB):
    lo = q[j-1] if j else -9.9; hi = q[j] if j < NB - 1 else 9.9
    print(f'  AOV_hat [{lo:5.2f},{hi:5.2f})  lam {lam_b[j]:+.3f}', flush=True)
print(f'единый lam: {flat:+.4f}\n', flush=True)
print(f'{"якорь":>7}{"контроль":>11}{"lam единый":>13}{"вентиль":>11}{"Δ вентиля":>12}', flush=True)
for A in (348, 378):
    d = D[A]; dd = d['p0'] * (d['mf'] - d['mc'])
    k = np.clip(np.digitize(d['a'], q), 0, NB - 1)
    sh = lambda v: float((d['z'] - v).std())
    s0 = sh(d['p0'] * d['mc']); s1 = sh(d['p0'] * d['mc'] + flat * dd)
    s2 = sh(d['p0'] * d['mc'] + lam_b[k] * dd)
    print(f'{A:>7}{s0:>11.5f}{s0-s1:>+13.5f}{s0-s2:>+11.5f}{s1-s2:>+12.5f}', flush=True)
print('\nготово', flush=True)
