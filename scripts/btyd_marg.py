"""Маржинальный ресурс BTYD-признаков поверх сильной базы.

Standalone на 378 дал +0.00063 при обеих положительных парных
разностях. Но standalone трижды обманывал: длинный контекст, ансамбль
сидов, ортогональная цель. Меряем направление поверх стека.

Ансамбль как в бою: 6 LGB (240/300/365 x 42/7) и те же CatBoost,
веса 0.4/0.6. Направление d = z(+btyd) - z(без), затем C поверх
v28-аналога с ортогонализацией против найденных направлений.
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

A = 378; O = Path('artifacts/neural'); R = Path('artifacts/recon')
HIST = (240, 300, 365); SEEDS = (42, 7); W_LGB, W_CB = 0.4, 0.6
df = load_panel(); acc = {k: [] for k in ('old', 'new')}
for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
    NEW = None
    for a in sorted(set(aid)):
        m = aid == a
        B, _ = btyd_feats(int(a), uid_tr[m])
        if NEW is None: NEW = np.zeros((len(y), B.shape[1]), dtype='float32')
        NEW[m] = B
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    uid = val.X['user_id'].to_numpy(); Bva, _ = btyd_feats(A, uid)
    nm = ['btyd_p_alive', 'btyd_en30', 'btyd_aov', 'btyd_gmv30']
    X2 = np.hstack([X, NEW]); Xva2 = np.hstack([Xva, Bva]); f2 = feats + nm
    print(f'=== история {h} ===', flush=True)
    for s in SEEDS:
        for fam in ('lgb', 'cb'):
            for tag, (Xt, Xv, ff) in (('old', (X, Xva, feats)), ('new', (X2, Xva2, f2))):
                t0 = time.perf_counter()
                M = (HurdleGBDT(config=ModelConfig(seed=s)) if fam == 'lgb'
                     else HurdleCatBoost(config=CatBoostConfig(seed=s)))
                M.fit(Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
                p, m_ = M.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
                acc[tag].append((fam, uid, np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))))
                print(f'  {fam} {tag} сид {s}: {time.perf_counter()-t0:.0f}с', flush=True)
    del X, X2, Xva, Xva2; gc.collect()

G = np.load(O / 'life_geom_a378.npz'); ref = G['user_id']; d_life = G['d_life']
E = np.load(R / 'cb_ens2.npz'); dann = np.load(R / 'd_annual_378.npy')
d16 = np.load(O / 'dz_a378.npz')['dz']; o = np.load(O / f'oofpm_a{A}.npz')
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(pl.DataFrame({'user_id': u_, 'v': np.asarray(v_, 'float64')}),
                             on='user_id', how='left')['v'].to_numpy()
zt = al(o['user_id'], np.log1p(o['y']))
def mix(tag):
    L = np.mean([al(u, v) for f, u, v in acc[tag] if f == 'lgb'], 0)
    C_ = np.mean([al(u, v) for f, u, v in acc[tag] if f == 'cb'], 0)
    return W_LGB * L + W_CB * C_
zo_, zn_ = mix('old'), mix('new')
d = zn_ - zo_
zl = np.mean([E[k] for k in E.files if k.startswith('lgb')], 0)
zc = np.mean([E[k] for k in E.files if k.startswith(('cb_', 'brd', 'dpw'))], 0)
cen = lambda v: v - v.mean()
z28 = 0.4 * zl + 0.6 * zc + 0.35 * cen(d16) + 0.0104 * cen(dann) + 0.24 * cen(d_life)
e = zt - z28
print(f'\nансамбль: без btyd {float((zt-zo_).std()):.5f} · с btyd {float((zt-zn_).std()):.5f}'
      f' · Δ {float((zt-zo_).std())-float((zt-zn_).std()):+.5f}')
print(f'shape v28-аналога {e.std():.5f}')
D3 = np.column_stack([cen(d16), cen(dann), cen(d_life)])
dd = cen(d); b = np.linalg.lstsq(D3, dd, rcond=None)[0]; dp = dd - D3 @ b
for lab, v in (('сырое', dd), ('ОРТОГОНАЛЬНОЕ', dp)):
    C_ = float((cen(e) * cen(v)).mean()); V = float((cen(v) ** 2).mean())
    a_ = C_ / max(V, 1e-15); g = C_ * C_ / max(V, 1e-15)
    new = np.sqrt(max(e.var() - g, 0))
    print(f'  {lab:<14} std {v.std():.5f} · C {C_:+.3e} · alpha* {a_:+.4f} · '
          f'прирост {e.std()-new:+.6f}')
print(f'  ортогональная доля {dp.std()/dd.std():.4f}')
np.savez_compressed(O / 'btyd_dir_a378.npz', user_id=ref, d=dd, d_orth=dp)
