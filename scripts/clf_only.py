"""Специализированный ансамбль классификаторов с metric-aware весом.

Отличие от классификаторов внутри hurdle — вес m^2: ошибка вероятности
у пользователя с большим чеком дороже для RMSLE, потому что
e_p^2 = m^2 (I - p)^2. Байесовский оптимум это не меняет (m зависит
только от X), но меняет, куда ограниченное дерево тратит ёмкость.

Проверяется не AUC, а конечный RMSLE после подстановки в v23:
z = p_new * m_old + те же поправки с теми же коэффициентами.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor, hurdle_glue,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.directions import marginal_gain
from catboost import CatBoostClassifier, Pool
df = load_panel(); O = Path('artifacts/neural')
SEEDS = (42, 7, 2026)
lg = lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))

def parts(anchor, stride, n_anch, caps=(200,100)):
    """Обучить hurdle и вернуть (p, m) на целевом якоре — для old-частей."""
    from ecup import HurdleGBDT
    from ecup.catboost_model import HurdleCatBoost, CatBoostConfig
    sp = SplitConfig(max_history=300, n_train_anchors=n_anch, stride=stride,
                     with_state=True, val_anchor=anchor) if False else None
    raise NotImplementedError

def build(anchor, stride, n_anch):
    sp = SplitConfig(max_history=300, n_train_anchors=n_anch, stride=stride, with_state=True)
    an = sp.train_anchors()
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    uid = Xd['user_id'].to_numpy(); X, feats = to_matrix(Xd); del Xd; gc.collect()
    ci, _ = anchor_offsets(aid, lv); wa = anchor_weights(aid)
    mhat = np.ones(len(y))
    for A in an:
        mk = aid == A
        f = O/f'oofpm_a{A}.npz'
        if f.exists():
            o = np.load(f); j = np.searchsorted(o['user_id'], uid[mk])
            j = np.clip(j, 0, len(o['user_id'])-1)
            ok = o['user_id'][j] == uid[mk]
            v = o['m0'][j]; v[~ok] = np.median(o['m0'])
            mhat[mk] = v
    val = build_anchor(df, anchor, sp, None); Xva, _ = to_matrix(val.X, feats)
    return X, (y>0).astype(int), wa*mhat**2, ci, Xva, val, lv[max(an)]

for ANCH in (348, 378):
    print(f'\n=== якорь {ANCH} ===', flush=True)
    ps = []
    for stride, n_a, fam in ((30,6,'lgb'), (10,18,'lgb'), (30,6,'cb')):
        X, I, wgt, ci, Xva, val, last = build(ANCH, stride, n_a)
        for s in SEEDS:
            t0 = time.perf_counter()
            if fam == 'lgb':
                c = lgb.LGBMClassifier(random_state=s, **ModelConfig().clf_params)
                c.set_params(n_estimators=200)
                c.fit(X, I, sample_weight=wgt, init_score=ci)
                raw = c.booster_.predict(Xva, raw_score=True)
            else:
                from ecup.catboost_model import CatBoostConfig
                cp = dict(CatBoostConfig().clf_params); cp['iterations'] = 1150
                c = CatBoostClassifier(random_seed=s, **cp)
                c.fit(Pool(X, label=I, weight=wgt, baseline=ci), verbose=False)
                raw = c.predict(Xva, prediction_type='RawFormulaVal')
            lp = float(np.log(last.p_bar/(1-last.p_bar)))
            ps.append(1/(1+np.exp(-(raw+lp))))
            print(f'  {fam} stride {stride} сид {s}: {time.perf_counter()-t0:.0f}с', flush=True)
            del c; gc.collect()
        del X, Xva; gc.collect()
    p_new = np.mean(ps, 0)
    np.savez_compressed(O/f'pclf_{ANCH}.npz', p=p_new, uid=val.X['user_id'].to_numpy())
    o = np.load(O/f'oofpm_a{ANCH}.npz'); z = np.log1p(o['y'])
    p_old, m_old = o['p0'], o['m0']
    print(f'  старый p*m: {(z-np.log1p(hurdle_glue(p_old,m_old))).std():.5f}')
    print(f'  новый p*m:  {(z-np.log1p(hurdle_glue(p_new,m_old))).std():.5f}')
    d_p = np.log1p(hurdle_glue(p_new, m_old)) - np.log1p(hurdle_glue(p_old, m_old))
    ex = []
    if ANCH == 378:
        d16 = np.load(O/'dz_a378.npz')['dz']; dann = np.load('/tmp/d_annual_378.npy')
        E = np.load('/tmp/cb_ens2.npz')
        zl = np.mean([E[k] for k in E.files if k.startswith('lgb')],0)
        zc = np.mean([E[k] for k in E.files if k.startswith(('cb_','brd','dpw'))],0)
        base = 0.4*zl+0.6*zc+0.35*(d16-d16.mean()); ex = [d16, dann]
    else:
        base = np.log1p(hurdle_glue(p_old, m_old))
    r = marginal_gain(z - base, d_p, existing=ex)
    print(f'  направление d_p поверх {"v23-аналога" if ex else "пары"}: '
          f'alpha {r["alpha_signed"]:+.4f} · маржинально {r["gain_marginal"]:+.5f}', flush=True)
