"""Предсказуема ли ошибка детекции и ошибка величины из имеющихся признаков.

Калибровка отвечает на вопрос E[I|p̂] = p̂. Нам нужен другой: осталась ли
в X информация об I, которую p̂ не использует. Это разные вещи — идеально
откалиброванный константный классификатор может быть очень плохим.

Точное разложение hurdle-ошибки, поэкземплярное:
    e = (I - p̂)·m̂ + I·(A - m̂) = e_p + e_m
Взаимодействие делится симметрично, доли складываются в полную MSE.

Затем два зонда. Для детекции модель учит (I - p̂) с весом m̂²: ошибка
вероятности у пользователя с большим чеком дороже для метрики. Для
величины — (A - m̂) только по покупавшим.
"""
import sys, warnings, gc; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb
from ecup import (SplitConfig, load_panel, build_training_set, to_matrix, build_anchor)
from ecup.directions import marginal_gain

print('=== точное разложение ошибки hurdle-пары ===')
for A in (348, 378):
    o = np.load(f'artifacts/neural/oofpm_a{A}.npz')
    y = o['y']; z = np.log1p(y); I = (y>0).astype(float); p, m = o['p0'], o['m0']
    ep = (I - p)*m; em = I*(z - m)
    M = float(np.mean((ep+em)**2))
    Ep, Em, X = float(np.mean(ep**2)), float(np.mean(em**2)), float(np.mean(ep*em))
    print(f'  якорь {A}: E[e_p²] {Ep:.4f} · E[e_m²] {Em:.4f} · 2E[e_p·e_m] {2*X:+.4f}')
    print(f'    доли: детекция {(Ep+X)/M:.1%} · величина {(Em+X)/M:.1%}')

print('\n=== зонды: предсказуема ли остаточная ошибка ===')
df = load_panel()
sp = SplitConfig(max_history=300, n_train_anchors=6, with_state=True)
def load(anchors):
    Xd, y, aid, lv = build_training_set(df, anchors, sp, None, verbose=False)
    uid = Xd['user_id'].to_numpy(); X, feats = to_matrix(Xd); del Xd; gc.collect()
    return X, y, uid, aid, feats
def oof(A, uid):
    o = np.load(f'artifacts/neural/oofpm_a{A}.npz')
    idx = np.searchsorted(o['user_id'], uid)
    ok = (idx < len(o['user_id'])) & (o['user_id'][np.clip(idx,0,len(o['user_id'])-1)] == uid)
    return {k: o[k][np.clip(idx,0,len(o['user_id'])-1)] for k in ('y','p0','m0')}, ok

d16 = np.load('artifacts/neural/dz_a378.npz')['dz']
dann = np.load('/tmp/d_annual_378.npy')
E = np.load('/tmp/cb_ens2.npz')
zl = np.mean([E[k] for k in E.files if k.startswith('lgb')],0)
zc = np.mean([E[k] for k in E.files if k.startswith(('cb_','brd','dpw'))],0)
PAR = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=500,
           reg_lambda=10.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
           verbose=-1, n_jobs=-1)
print(f'{"зонд":<12} {"фолд":<16} {"alpha":>9} {"маржин":>10}')
for tr, te in (([228,258,288], 348), ([258,288,318], 378)):
    Xtr, ytr, uid_tr, aid, feats = load(tr)
    o_tr, ok = oof(0, uid_tr) if False else (None, None)
    P, M_, Yv = [], [], []
    for A in tr:
        mk = aid == A
        d, _ = oof(A, uid_tr[mk]); P.append(d['p0']); M_.append(d['m0']); Yv.append(d['y'])
    p_tr = np.concatenate(P); m_tr = np.concatenate(M_); y_tr = np.concatenate(Yv)
    ordr = np.argsort(np.concatenate([np.where(aid==A)[0] for A in tr]))
    p_tr, m_tr, y_tr = p_tr[ordr], m_tr[ordr], y_tr[ordr]
    I_tr = (y_tr > 0).astype(float); z_tr = np.log1p(y_tr)
    val = build_anchor(df, te, sp, None); Xte, _ = to_matrix(val.X, feats)
    ote = np.load(f'artifacts/neural/oofpm_a{te}.npz')
    p_te, m_te = ote['p0'], ote['m0']; z_te = np.log1p(ote['y'])
    base = (0.4*zl+0.6*zc+0.35*(d16-d16.mean())) if te == 378 else p_te*m_te
    e_te = z_te - base
    ex = [d16, dann] if te == 378 else []
    # зонд детекции
    g = lgb.LGBMRegressor(**PAR).fit(Xtr, I_tr - p_tr, sample_weight=m_tr**2)
    d_p = m_te * g.predict(Xte)
    r = marginal_gain(e_te, d_p, existing=ex)
    print(f'  {"детекция":<10} {str(tr[-1])+" -> "+str(te):<16} '
          f'{r["alpha_signed"]:>+9.4f} {r["gain_marginal"]:>+10.5f}')
    # зонд величины
    pos = I_tr > 0
    g2 = lgb.LGBMRegressor(**PAR).fit(Xtr[pos], (z_tr - m_tr)[pos])
    d_m = p_te * g2.predict(Xte)
    r2 = marginal_gain(e_te, d_m, existing=ex)
    print(f'  {"величина":<10} {str(tr[-1])+" -> "+str(te):<16} '
          f'{r2["alpha_signed"]:>+9.4f} {r2["gain_marginal"]:>+10.5f}')
    del Xtr, Xte; gc.collect()
