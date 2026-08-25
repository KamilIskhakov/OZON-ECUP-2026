"""Сверка d0 против d18: меняет ли ручка a_m геометрию направления.

Валидация BTYD шла при m_offset = l_plus, а боевой прогон использует
l_plus + 0.18. Поскольку склейка даёт z = p*m, то

    d_18 - d_0 = 0.18 (p_1 - p_0),

и величина зависит от того, насколько BTYD-признаки меняют
КЛАССИФИКАТОР. У BTYD основной вклад ожидается в частотную часть,
поэтому проверить надо.

Решает не сырая корреляция, а ортогональная: в сабмит идёт остаток
после проекции на старые направления, и малое изменение dp может почти
не тронуть raw, но сдвинуть маленькую ортогональную компоненту.

Одна история (300) и два сида — достаточно для оценки геометрии.
Пороги заданы заранее: rho_perp > 0.98 и 0.9 < s_perp < 1.1 —
амплитуду не трогаем; rho_perp < 0.95 — probe с половинной амплитудой.
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
from strong_base import annual

O = Path('artifacts/neural'); H = 300; SEEDS = (42, 7); A_M = 0.18
W_LGB, W_CB = 0.4, 0.6
df = load_panel()
FIN = SplitConfig(max_history=300, with_state=True).final_anchor
sp = SplitConfig(max_history=H, with_state=True); an = sp.refit_anchors()
Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
NEW = None
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
acc = {}
for s in SEEDS:
    for fam in ('lgb', 'cb'):
        for tag, (Xt, Xv, ff) in (('old', (X, Xte, feats)), ('new', (X2, Xte2, f2))):
            t0 = time.perf_counter()
            M = (HurdleGBDT(config=ModelConfig(seed=s)) if fam == 'lgb'
                 else HurdleCatBoost(config=CatBoostConfig(seed=s)))
            M.fit(Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
            p, m_ = M.predict_parts(Xv, p_target=last.p_bar, m_offset=0.0)
            # z = p*(m + c): оба смещения из ОДНИХ предсказаний, без переобучения
            for c, lab in ((0.0, 'd0'), (A_M, 'd18')):
                z_ = np.log1p(hurdle_glue(p, np.clip(m_ + last.l_plus + c, 0, None)))
                acc.setdefault((fam, tag, lab), []).append((uid, z_))
            print(f'  {fam} {tag} сид {s}: {time.perf_counter()-t0:.0f}с', flush=True)
ref = np.load(O / 'dz_prod_a408.npz')['user_id']
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(pl.DataFrame({'user_id': u_, 'z': v_}),
                             on='user_id', how='left')['z'].to_numpy()
cen = lambda v: v - v.mean()
d = {}
for lab in ('d0', 'd18'):
    L = np.mean([al(*v) for v in acc[('lgb', 'new', lab)]], 0) - \
        np.mean([al(*v) for v in acc[('lgb', 'old', lab)]], 0)
    C_ = np.mean([al(*v) for v in acc[('cb', 'new', lab)]], 0) - \
         np.mean([al(*v) for v in acc[('cb', 'old', lab)]], 0)
    d[lab] = cen(W_LGB * L + W_CB * C_)
d_gru = np.nan_to_num(al(ref, np.load(O / 'dz_prod_a408.npz')['dz']))
lm = np.load(O / 'longmoney_prod_a408.npz'); d_life = np.nan_to_num(al(lm['user_id'], lm['d']))
gp = np.load(O / 'gruprod_dir_a408.npz'); d_new = np.nan_to_num(al(gp['user_id'], gp['d_raw']))
D = np.column_stack([cen(d_gru), cen(annual(df, FIN, ref)), cen(d_life), cen(d_new)])
perp = {}
for lab in d:
    b = np.linalg.lstsq(D, d[lab], rcond=None)[0]
    perp[lab] = d[lab] - D @ b
r = float(np.corrcoef(d['d0'], d['d18'])[0, 1])
s = float(d['d18'].std() / d['d0'].std())
rp = float(np.corrcoef(perp['d0'], perp['d18'])[0, 1])
sp_ = float(perp['d18'].std() / perp['d0'].std())
print(f'\nсырое:         rho {r:.4f} · s {s:.4f}')
print(f'ОРТОГОНАЛЬНОЕ: rho {rp:.4f} · s {sp_:.4f}')
print(f'ортогональная доля: d0 {perp["d0"].std()/d["d0"].std():.3f} · '
      f'd18 {perp["d18"].std()/d["d18"].std():.3f}')
v = 'амплитуду не трогаем' if (rp > 0.98 and 0.9 < sp_ < 1.1) else (
    'probe с половинной амплитудой' if rp < 0.95 else 'сохранить амплитуду 0.0108')
print(f'\nрешение по заданным заранее порогам: {v}')
