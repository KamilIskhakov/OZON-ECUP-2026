"""Сколько ошибки детекции — промах классификатора, а сколько бернуллиев шум.

Если p = P(I=1|X) верна, то E[e_p^2] = E[m^2 p(1-p)] по построению:
это дисперсия самого бинарного исхода, неустранимая никакой моделью.
Разрыв между наблюдаемым и этим полом и есть запас классификатора.

Зонд симметричен на обоих якорях: измеряется поверх ТОЙ ЖЕ пары p*m,
относительно которой построены цели. Прежний замер сравнивал 348 с парой,
а 378 с ансамблем — два разных вопроса под одним именем.

Параметризация через offset к логиту, а не разность вероятностей:
p_new = sigmoid(logit(p0) + f(X)), вес m^2 — именно та величина, что
входит в e_p^2 = m^2 (I - p)^2.
"""
import sys, warnings, gc; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb
from ecup import SplitConfig, load_panel, build_training_set, to_matrix, build_anchor
from ecup.directions import marginal_gain

print('=== бернуллиев пол детекции ===')
for A in (318, 348, 378):
    o = np.load(f'artifacts/neural/oofpm_a{A}.npz')
    y = o['y']; I = (y>0).astype(float); p, m = o['p0'], o['m0']; z = np.log1p(y)
    ep2 = float(np.mean(m**2*(I-p)**2)); floor = float(np.mean(m**2*p*(1-p)))
    em2 = float(np.mean(I*(z-m)**2)); cross = float(2*np.mean((I-p)*m*I*(z-m)))
    print(f'  якорь {A}: E[e_p²] {ep2:.4f} · пол E[m²p(1-p)] {floor:.4f} · '
          f'запас {ep2-floor:+.4f} ({100*(ep2-floor)/ep2:+.1f}%)')
    print(f'    E[e_m²] {em2:.4f} · 2E[e_p·e_m] {cross:+.4f}')

print('\n=== зонд с offset к логиту, симметрично на обоих якорях ===')
df = load_panel(); sp = SplitConfig(max_history=300, n_train_anchors=6, with_state=True)
PAR = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=500,
           reg_lambda=10.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
           verbose=-1, n_jobs=-1)
lg = lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))
print(f'{"зонд":<12} {"фолд":<18} {"alpha":>9} {"маржин":>10}')
for tr, te in (([228,258,288], 348), ([258,288,318], 378)):
    Xd, y_all, aid, lv = build_training_set(df, tr, sp, None, verbose=False)
    uid = Xd['user_id'].to_numpy(); X, feats = to_matrix(Xd); del Xd; gc.collect()
    P, M_, Y = np.empty(len(uid)), np.empty(len(uid)), np.empty(len(uid))
    for A in tr:
        mk = aid == A
        o = np.load(f'artifacts/neural/oofpm_a{A}.npz')
        j = np.searchsorted(o['user_id'], uid[mk])
        P[mk], M_[mk], Y[mk] = o['p0'][j], o['m0'][j], o['y'][j]
    I = (Y>0).astype(int); z = np.log1p(Y)
    val = build_anchor(df, te, sp, None); Xte, _ = to_matrix(val.X, feats)
    ot = np.load(f'artifacts/neural/oofpm_a{te}.npz')
    p_t, m_t, z_t = ot['p0'], ot['m0'], np.log1p(ot['y'])
    e = z_t - p_t*m_t                                  # ОДИН И ТОТ ЖЕ объект на обоих
    c = lgb.LGBMClassifier(**PAR)
    c.fit(X, I, sample_weight=M_**2, init_score=lg(P))
    raw = c.booster_.predict(Xte, raw_score=True)
    d_p = m_t*(1/(1+np.exp(-(lg(p_t)+raw))) - p_t)
    r = marginal_gain(e, d_p, existing=[])
    print(f'  {"детекция":<10} {str(tr[-1])+" -> "+str(te):<18} '
          f'{r["alpha_signed"]:>+9.4f} {r["gain_marginal"]:>+10.5f}')
    pos = I > 0
    g = lgb.LGBMRegressor(**PAR).fit(X[pos], (z-M_)[pos])
    d_m = p_t*g.predict(Xte)
    r2 = marginal_gain(e, d_m, existing=[])
    print(f'  {"величина":<10} {str(tr[-1])+" -> "+str(te):<18} '
          f'{r2["alpha_signed"]:>+9.4f} {r2["gain_marginal"]:>+10.5f}')
    del X, Xte; gc.collect()
