"""Отдельная ветка по дальней истории: маленький GBDT на остатке.

Годовой фильтр показал, что незабранный сигнал лежит в дальней временной
структуре. Вместо ручного подбора очередной формы даём деревьям пакет
низкочастотных признаков и позволяем найти взаимодействия с типом
пользователя — именно они объясняли бы, почему сила годового эффекта
меняется между якорями.

Ёмкость сознательно мала: задача не построить универсальную модель,
а объяснить остаток того, чего боевой ансамбль не видит.
"""
from __future__ import annotations
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from ecup import load_panel
from ecup.directions import marginal_gain
import lightgbm as lgb

df = load_panel()
K = 11                                   # блоков по 30 дней от прошлогоднего окна

def blocks(A, uid):
    u = pl.Series('user_id', uid); out = {k: [] for k in ('g','n','b','s')}
    for j in range(K):
        lo, hi = A-364+30*j, A-335+30*j
        if lo < 0:
            for k in out: out[k].append(np.zeros(len(uid)))
            continue
        t=(df.filter(pl.col('d').is_between(lo,hi)&pl.col('user_id').is_in(u))
             .group_by('user_id').agg(g=pl.col('gmv').sum(), n=pl.col('to_ord').sum(),
                                      b=(pl.col('gmv')>0).sum(), s=pl.col('searches').sum()))
        k_=(pl.DataFrame({'user_id':uid}).join(t,on='user_id',how='left')
              .with_columns(pl.exclude('user_id').fill_null(0)).sort('user_id'))
        for k in out: out[k].append(np.log1p(k_[k].to_numpy().astype('float64')))
    return {k: np.column_stack(v) for k, v in out.items()}

def feats(A, uid):
    B = blocks(A, uid); G = B['g']
    j = np.arange(K, dtype=float)
    ph = 2*np.pi*(-379.5 + 30*j)/365
    W = {'ann': np.cos(ph), 'sin': np.sin(ph), 'quad': (j-j.mean())**2,
         'lin': j-j.mean(), 'ann2': np.cos(2*ph)}
    F, N = [], []
    for nm, w in W.items():
        wc = w - w.mean()
        for q in ('g','n','b'):
            F.append(B[q] @ wc); N.append(f'{nm}_{q}')
    F += [G.mean(1), G.std(1), G.max(1)-G.min(1),
          G[:, :3].mean(1)-G[:, 4:7].mean(1), G[:, 4:7].mean(1)-G[:, 8:].mean(1),
          (G > 0).sum(1).astype(float), np.polyfit(np.arange(3.), G[:, -3:].T, 1)[0],
          np.polyfit(np.arange(6.), G[:, -6:].T, 1)[0],
          B['s'].mean(1), B['n'].mean(1), B['b'].mean(1)]
    N += ['g_mean','g_std','g_range','early_mid','mid_late','nonzero',
          'slope3','slope6','s_mean','n_mean','b_mean']
    return np.column_stack(F).astype('float32'), N

FOLDS = [([228,258,288], 348), ([258,288,318], 378)]
d16 = np.load('artifacts/neural/dz_a378.npz')['dz']
E = np.load('/tmp/cb_ens2.npz')
zl = np.mean([E[k] for k in E.files if k.startswith('lgb')],0)
zc = np.mean([E[k] for k in E.files if k.startswith(('cb_','brd','dpw'))],0)
res = []
for tr, te in FOLDS:
    Xs, ys = [], []
    for A in tr:
        o = np.load(f'artifacts/neural/oof_a{A}.npz')
        X, names = feats(A, o['user_id'])
        Xs.append(X); ys.append(np.log1p(o['y']) - o['z0'])
    ot = np.load(f'artifacts/neural/oof_a{te}.npz')
    Xt, _ = feats(te, ot['user_id'])
    m = lgb.LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.03,
                          min_child_samples=2000, reg_lambda=20.0,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          verbose=-1, n_jobs=-1)
    m.fit(np.vstack(Xs), np.concatenate(ys))
    d = m.predict(Xt)
    e = np.log1p(ot['y']) - ot['z0']
    ex = [d16, np.load('/tmp/d_annual_378.npy')] if te == 378 else []
    r = marginal_gain(e, d, existing=ex)
    res.append(r)
    print(f'обучение {tr} -> оценка {te}: alpha {r["alpha_signed"]:+.5f} · '
          f'сольно {r["gain_solo"]:+.5f} · маржинально {r["gain_marginal"]:+.5f}', flush=True)
    if te == 378:
        imp = sorted(zip(names, m.booster_.feature_importance('gain')),
                     key=lambda x: -x[1])[:8]
        print('  вклад признаков:', ', '.join(f'{n}' for n, _ in imp))
sg = [np.sign(r['alpha_signed']) for r in res]
print(f'\nзнаки: {[int(x) for x in sg]} — {"совпадают" if len(set(sg))==1 else "РАЗНЫЕ"}')
