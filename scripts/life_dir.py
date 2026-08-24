"""Направление блока беcкэповой истории на заданном якоре, честно.

d_life(A) = 0.4 (z_LGB^{+блок} - z_LGB) + 0.6 (z_CB^{+блок} - z_CB),
обучение строго на якорях a + 30 <= A. Нужно для сборки сильной базы
на 318 и 348 (на 378 уже посчитано в life_geom_a378.npz).
"""
import sys, warnings, gc, time, os; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from ecup.catboost_model import CatBoostConfig, HurdleCatBoost
from ecup.market import market_features, _market

O = Path('artifacts/neural'); HIST = (240, 300, 365); SEEDS = (42, 7)
W_LGB, W_CB = 0.4, 0.6
A = int(os.environ['ANCHOR'])
df = load_panel(); mkt = _market(df)
acc = {k: [] for k in ('lgb_old', 'lgb_new', 'cb_old', 'cb_new')}
for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
    NEW = None
    for a in sorted(set(aid)):
        m = aid == a
        B, nm = market_features(df, int(a), uid_tr[m], mkt)
        if NEW is None:
            keep = [i for i, c in enumerate(nm) if c.startswith('life_')]
            nm_l = [nm[i] for i in keep]
            NEW = np.zeros((len(y), len(keep)), dtype='float32')
        NEW[m] = B[:, keep]
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    uid = val.X['user_id'].to_numpy()
    Bva, _ = market_features(df, A, uid, mkt)
    X2 = np.hstack([X, NEW]); Xva2 = np.hstack([Xva, Bva[:, keep]]); f2 = feats + nm_l
    print(f'=== якорь {A} · история {h} · обучающие {an} ===', flush=True)
    for s in SEEDS:
        for fam in ('lgb', 'cb'):
            for tag, (Xt, Xv, ff) in (('old', (X, Xva, feats)), ('new', (X2, Xva2, f2))):
                t0 = time.perf_counter()
                if fam == 'lgb':
                    hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
                        Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
                else:
                    hm = HurdleCatBoost(config=CatBoostConfig(seed=s)).fit(
                        Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
                p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
                acc[f'{fam}_{tag}'].append(
                    (uid, np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))))
                print(f'  {fam} {tag} сид {s}: {time.perf_counter()-t0:.0f}с', flush=True)
    del X, X2, Xva, Xva2; gc.collect()

o = np.load(O / f'oofpm_a{A}.npz'); ref = o['user_id']
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(pl.DataFrame({'user_id': u_, 'z': v_}),
                             on='user_id', how='left')['z'].to_numpy()
Z = {k: np.mean([al(*v) for v in lst], 0) for k, lst in acc.items()}
d = W_LGB * (Z['lgb_new'] - Z['lgb_old']) + W_CB * (Z['cb_new'] - Z['cb_old'])
np.savez_compressed(O / f'life_dir_a{A}.npz', user_id=ref, d_life=d)
print(f'\nсохранено life_dir_a{A}.npz · std {d.std():.5f} · среднее {d.mean():+.5f}',
      flush=True)
