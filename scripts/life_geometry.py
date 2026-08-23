"""Сколько от life остаётся ПОСЛЕ полного v23.

Вопрос не «насколько life улучшает одиночный LGB», а сколько его
ресурса переживает CatBoost, Gap-GRU и годовой фильтр — они могли
чинить ту же ошибку.

Направление считается на том же ансамбле и с теми же весами, что в
боевом рецепте:

    d_life = 0.4 (z_lgb_new - z_lgb_old) + 0.6 (z_cb_new - z_cb_old)

База — реконструкция точного v23 на якоре 378:

    z_v23 = 0.4 zl_saved + 0.6 zc + 0.35 (d16 - mean) + 0.0104 (dann - mean)

Затем ортогонализация против уже найденных детерминированных
направлений D = [d_GRU, d_annual] и обычные

    C = Cov(e, d_perp),  V = Var(d_perp),  alpha* = C/V,  dM = C^2/V
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from ecup.catboost_model import CatBoostConfig, HurdleCatBoost
from ecup.market import market_features, rate_features, _market

import os
BLOCK = os.environ.get('BLOCK', 'life')
A = 378; O = Path('artifacts/neural'); HIST = (240, 300, 365); SEEDS = (42, 7)
W_LGB, W_CB = 0.4, 0.6
df = load_panel(); mkt = _market(df)
acc = {k: [] for k in ('lgb_old', 'lgb_new', 'cb_old', 'cb_new')}

for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
    def block(a_, u_):
        if BLOCK == 'life':
            B_, nm_ = market_features(df, a_, u_, mkt)
            k_ = [i for i, c in enumerate(nm_) if c.startswith('life_')]
            return B_[:, k_], [nm_[i] for i in k_]
        return rate_features(df, a_, u_, scopes=('full',))
    NEW = None
    for a in sorted(set(aid)):
        m = aid == a
        B, nm_l = block(int(a), uid_tr[m])
        if NEW is None:
            NEW = np.zeros((len(y), B.shape[1]), dtype='float32')
        NEW[m] = B
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    uid = val.X['user_id'].to_numpy()
    Bva, _ = block(A, uid)
    X2 = np.hstack([X, NEW]); Xva2 = np.hstack([Xva, Bva]); f2 = feats + nm_l
    print(f'\n=== история {h} · якорей {len(an)} ===', flush=True)
    for s in SEEDS:
        for fam in ('lgb', 'cb'):
            for tag, (Xt, Xv, ff) in (('old', (X, Xva, feats)), ('new', (X2, Xva2, f2))):
                t0 = time.perf_counter()
                if fam == 'lgb':
                    mc = ModelConfig(seed=s)
                    hm = HurdleGBDT(config=mc).fit(Xt, y, feature_names=ff, sample_weight=w,
                                                   z_offset=zo, clf_init=ci)
                else:
                    mc = CatBoostConfig(seed=s)
                    hm = HurdleCatBoost(config=mc).fit(Xt, y, feature_names=ff, sample_weight=w,
                                                       z_offset=zo, clf_init=ci)
                z_ = np.log1p(hm.predict(Xv, p_target=last.p_bar, m_offset=last.l_plus))
                acc[f'{fam}_{tag}'].append((uid, z_))
                print(f'  {fam} {tag} сид {s}: {time.perf_counter()-t0:.0f}с', flush=True)
    del X, X2, Xva, Xva2; gc.collect()

o = np.load(O / f'oofpm_a{A}.npz'); ref = o['user_id']; z = np.log1p(o['y'])
def align(lst):
    out = []
    for u_, v_ in lst:
        t = pl.DataFrame({'user_id': u_, 'z': v_})
        out.append(pl.DataFrame({'user_id': ref}).join(t, on='user_id', how='left')['z'].to_numpy())
    return np.mean(out, 0)
Z = {k: align(v) for k, v in acc.items()}
d_life = W_LGB * (Z['lgb_new'] - Z['lgb_old']) + W_CB * (Z['cb_new'] - Z['cb_old'])

E = np.load('/tmp/cb_ens2.npz')
zl_saved = np.mean([E[k] for k in E.files if k.startswith('lgb')], 0)
zc = np.mean([E[k] for k in E.files if k.startswith(('cb_', 'brd', 'dpw'))], 0)
d16 = np.load(O / f'dz_a{A}.npz')['dz']; dann = np.load(f'/tmp/d_annual_{A}.npy')
CORR = 0.35 * (d16 - d16.mean()) + 0.0104 * (dann - dann.mean())
z_v23 = 0.4 * zl_saved + 0.6 * zc + CORR
e = z - z_v23
print(f'\nБЛОК {BLOCK} · shape точного v23 на {A}: {e.std():.5f}', flush=True)

D = np.column_stack([d16 - d16.mean(), dann - dann.mean()])
dl = d_life - d_life.mean()
beta = np.linalg.lstsq(D, dl, rcond=None)[0]
dp = dl - D @ beta
for nm_, v in (('d_life сырое', dl), ('d_life ортогональное', dp)):
    C = float(((e - e.mean()) * (v - v.mean())).mean()); V = float(((v - v.mean())**2).mean())
    a = C / max(V, 1e-15); dM = C * C / max(V, 1e-15)
    new = np.sqrt(max(e.var() - dM, 0))
    print(f'  {nm_:<22} std {v.std():.5f} · C {C:+.3e} · V {V:.3e} · '
          f'alpha* {a:+.4f} · ΔM {dM:.3e} · shape {e.std():.5f} -> {new:.5f} '
          f'({e.std()-new:+.5f})', flush=True)
print(f'  ортогональная доля {dp.std()/dl.std():.4f} · '
      f'corr(d_life, GRU) {np.corrcoef(dl, d16)[0,1]:+.4f} · '
      f'corr(d_life, annual) {np.corrcoef(dl, dann)[0,1]:+.4f}')
print(f'\nпри применении с alpha = 1: shape '
      f'{float((e - dl).std()):.5f} против {e.std():.5f}', flush=True)
np.savez_compressed(O / f'{BLOCK}_geom_a{A}.npz', user_id=ref, d_life=d_life, e=e, z_v23=z_v23)
print('готово', flush=True)
