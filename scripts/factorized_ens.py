"""Ансамблировать головы отдельно или произведение целиком.

mean(p_k m_k) != mean(p_k) mean(m_k). Это не тонкость: детекция несёт
76-79% ошибки hurdle-пары, а ансамбль до сих пор строился на уровне
конечного z, то есть смешивал разнообразие двух разных задач в одной
операции усреднения.

Сохраняем части всех моделей и сравниваем три способа сборки.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, HurdleGBDT, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.catboost_model import HurdleCatBoost, CatBoostConfig
df = load_panel(); sp = SplitConfig(max_history=300, n_train_anchors=6, with_state=True)
an = sp.train_anchors()
Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
X, feats = to_matrix(Xd); del Xd; gc.collect()
w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
val = build_anchor(df, sp.val_anchor, sp, None); Xva, _ = to_matrix(val.X, feats)
z = np.log1p(val.y)
CFG = [('lgb', s, (200,100)) for s in (42,7,2026)] + \
      [('cb', s, (1150,700)) for s in (42,7,2026)]
Ps, Ms, Zs, fam = [], [], [], []
for f, s, caps in CFG:
    t0 = time.perf_counter()
    if f == 'lgb':
        cfg = ModelConfig(seed=s, early_stopping_rounds=None); cls = HurdleGBDT
        cfg.clf_params['n_estimators'], cfg.reg_params['n_estimators'] = caps
    else:
        cfg = CatBoostConfig(seed=s, early_stopping_rounds=None); cls = HurdleCatBoost
        cfg.clf_params['iterations'], cfg.reg_params['iterations'] = caps
    m_ = cls(config=cfg).fit(X, y, feature_names=feats, sample_weight=w,
                             z_offset=zo, clf_init=ci)
    p, mm = m_.predict_parts(Xva, p_target=last.p_bar, m_offset=last.l_plus)
    mm = np.clip(mm, 0, None)
    Ps.append(p); Ms.append(mm); Zs.append(np.log1p(hurdle_glue(p, mm))); fam.append(f)
    print(f'  {f} сид {s}: shape {(z-Zs[-1]).std():.5f} · {time.perf_counter()-t0:.0f}с',
          flush=True)
    del m_; gc.collect()
P, M, Z = np.array(Ps), np.array(Ms), np.array(Zs)
fam = np.array(fam); wf = np.where(fam=='lgb', 0.4/3, 0.6/3)
sh = lambda v: float((z-v).std())
z_prod = wf @ Z                                   # как сейчас: усреднение произведений
z_sep  = np.log1p(hurdle_glue(wf @ P, wf @ M))    # усреднение частей, потом склейка
print(f'\nусреднение произведений (как сейчас): {sh(z_prod):.5f}')
print(f'усреднение частей, затем склейка:     {sh(z_sep):.5f}  '
      f'({sh(z_prod)-sh(z_sep):+.5f})')
print('\nразные веса для голов (доля LightGBM):')
best = None
for ap_ in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    for bm in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        a_ = np.where(fam=='lgb', ap_/3, (1-ap_)/3)
        b_ = np.where(fam=='lgb', bm/3, (1-bm)/3)
        v = sh(np.log1p(hurdle_glue(a_ @ P, b_ @ M)))
        if best is None or v < best[0]: best = (v, ap_, bm)
    print('  p=%.1f: ' % ap_ + ' '.join(
        f'{sh(np.log1p(hurdle_glue(np.where(fam=="lgb",ap_/3,(1-ap_)/3) @ P, np.where(fam=="lgb",b/3,(1-b)/3) @ M))):.5f}'
        for b in (0.0,0.2,0.4,0.6,0.8,1.0)))
print(f'\nлучшее: доля LGB в p {best[1]:.1f}, в m {best[2]:.1f} -> {best[0]:.5f}')
np.savez_compressed(Path('artifacts/neural')/'parts_378.npz', P=P, M=M, Z=Z,
                    fam=fam, z=z, uid=val.X['user_id'].to_numpy())
