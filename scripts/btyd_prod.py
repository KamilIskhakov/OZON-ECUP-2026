"""Боевое направление BTYD-признаков на якоре 408.

Маржинал поверх полного стека v34 составил +0.000076 при ортогональной
доле 0.932 и corr с новым нейронаправлением всего +0.084 — BG/NBD
нашёл ось, независимую от всего найденного.

    d = 0.4 (z_LGB^{+btyd} - z_LGB) + 0.6 (z_CB^{+btyd} - z_CB)

Обучение на боевых якорях, признаки BTYD подгоняются на каждом якоре
отдельно и честно (используется только история до якоря).
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from ecup.catboost_model import CatBoostConfig, HurdleCatBoost
from btyd_feat import btyd_feats

O = Path('artifacts/neural'); HIST = (240, 300, 365); SEEDS = (42, 7)
W_LGB, W_CB = 0.4, 0.6; A_M = 0.18
df = load_panel()
FIN = SplitConfig(max_history=300, with_state=True).final_anchor
print(f'боевой якорь {FIN}', flush=True)
acc = {k: [] for k in ('lgb_old', 'lgb_new', 'cb_old', 'cb_new')}
for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
    an = sp.refit_anchors()
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
    t0 = time.perf_counter(); NEW = None
    for a in sorted(set(aid)):
        m = aid == a
        B, _ = btyd_feats(int(a), uid_tr[m])
        if NEW is None: NEW = np.zeros((len(y), B.shape[1]), dtype='float32')
        NEW[m] = B
    fin = build_anchor(df, FIN, sp, None, with_target=False)
    Xte, _ = to_matrix(fin.X, feats); uid = fin.X['user_id'].to_numpy()
    Bte, _ = btyd_feats(FIN, uid)
    nm = ['btyd_p_alive', 'btyd_en30', 'btyd_aov', 'btyd_gmv30']
    X2 = np.hstack([X, NEW]); Xte2 = np.hstack([Xte, Bte]); f2 = feats + nm
    p_base = last.p_bar; m_off = last.l_plus + A_M
    print(f'=== история {h} · якорей {len(an)} · признаки за '
          f'{time.perf_counter()-t0:.0f}с ===', flush=True)
    for s in SEEDS:
        for fam in ('lgb', 'cb'):
            for tag, (Xt, Xv, ff) in (('old', (X, Xte, feats)), ('new', (X2, Xte2, f2))):
                t0 = time.perf_counter()
                M = (HurdleGBDT(config=ModelConfig(seed=s)) if fam == 'lgb'
                     else HurdleCatBoost(config=CatBoostConfig(seed=s)))
                M.fit(Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
                z_ = np.log1p(M.predict(Xv, p_target=p_base, m_offset=m_off))
                acc[f'{fam}_{tag}'].append((uid, z_))
                print(f'  {fam} {tag} сид {s}: {time.perf_counter()-t0:.0f}с', flush=True)
    del X, X2, Xte, Xte2; gc.collect()

ref = np.load(O / 'dz_prod_a408.npz')['user_id']
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(pl.DataFrame({'user_id': u_, 'z': v_}),
                             on='user_id', how='left')['z'].to_numpy()
Z = {k: np.mean([al(*v) for v in lst], 0) for k, lst in acc.items()}
d = W_LGB * (Z['lgb_new'] - Z['lgb_old']) + W_CB * (Z['cb_new'] - Z['cb_old'])
cen = lambda v: v - v.mean()
d_gru = np.nan_to_num(al(ref, np.load(O / 'dz_prod_a408.npz')['dz']))
lm = np.load(O / 'longmoney_prod_a408.npz'); d_life = np.nan_to_num(al(lm['user_id'], lm['d']))
gp = np.load(O / 'gruprod_dir_a408.npz'); d_new = np.nan_to_num(al(gp['user_id'], gp['d_raw']))
from strong_base import annual
d_ann = annual(df, FIN, ref)
D = np.column_stack([cen(d_gru), cen(d_ann), cen(d_life), cen(d_new)])
b = np.linalg.lstsq(D, cen(d), rcond=None)[0]
dp = cen(d) - D @ b
print(f'\nнаправление: std {d.std():.5f} · ортогональное {dp.std():.5f} · '
      f'доля {dp.std()/cen(d).std():.3f}')
print(f'corr: GRU {np.corrcoef(d,d_gru)[0,1]:+.4f} · годовой {np.corrcoef(d,d_ann)[0,1]:+.4f} '
      f'· life {np.corrcoef(d,d_life)[0,1]:+.4f} · GRU-new {np.corrcoef(d,d_new)[0,1]:+.4f}')
np.savez_compressed(O / 'btyd_prod_a408.npz', user_id=ref, d=cen(d), d_orth=dp)
print('сохранено btyd_prod_a408.npz', flush=True)
