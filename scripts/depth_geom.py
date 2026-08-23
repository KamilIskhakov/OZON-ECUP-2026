"""Направление глубины ансамбля, измеренное ПОВЕРХ v28.

Мерить поверх v23 нельзя: v28 уже забрал 0.24 life-направления, а
life и depth исправляют один и тот же недостаток дальней истории.
Иначе один сигнал засчитается дважды.

    d_depth = z_LGB^{300,365,420} - z_LGB^{240,300,365}
    d_full  = 0.4 d_depth          (CatBoost не трогаем: для него
                                    глубину мы не валидировали)
    z_28    = z_v23 + 0.24 (d_life - mean d_life)
    e_28    = z - z_28

Ортогонализация против GRU/annual/life не нужна: они уже внутри z_28.
Корреляции печатаются только как диагностика.

Печатается И оракульный alpha*, И фактический результат при alpha = 1.
Урок v27: если alpha = 1 уже офлайн хуже, полная замена не переносится;
если лучше — это не произвольная поправка, а замена слабого участника
h=240 на h=420, и её можно ставить целиком.
"""
import sys, warnings, gc, time, os; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT

A = 378; O = Path('artifacts/neural'); SEEDS = (42, 7)
OLD, NEW = (240, 300, 365), (300, 365, 420)
HS = sorted(set(OLD) | set(NEW))
df = load_panel(); P = {}
for h in HS:
    sp = SplitConfig(max_history=h, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    uid = val.X['user_id'].to_numpy()
    for s in SEEDS:
        t0 = time.perf_counter()
        hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
            X, y, feature_names=feats, sample_weight=w, z_offset=zo, clf_init=ci)
        p, m_ = hm.predict_parts(Xva, p_target=last.p_bar, m_offset=last.l_plus)
        P[(h, s)] = (uid, np.log1p(hurdle_glue(p, np.clip(m_, 0, None))))
        print(f'  h={h} сид {s}: {time.perf_counter()-t0:.0f}с', flush=True)
    del X, Xva; gc.collect()

G = np.load(O / f'life_geom_a{A}.npz')
ref = G['user_id']; z_v23 = G['z_v23']; d_life = G['d_life']
o = np.load(O / f'oofpm_a{A}.npz')
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(pl.DataFrame({'user_id': u_, 'p': v_}),
                             on='user_id', how='left')['p'].to_numpy()
Z = {k: al(*v) for k, v in P.items()}
z = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y'])}).join(
    key, on='user_id', how='semi')
z = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y'])})
z = key.join(z, on='user_id', how='left')['z'].to_numpy()

zold = np.mean([Z[(h, s)] for h in OLD for s in SEEDS], 0)
znew = np.mean([Z[(h, s)] for h in NEW for s in SEEDS], 0)
d_depth = znew - zold; d_full = 0.4 * d_depth
dl = d_life - d_life.mean()
z28 = z_v23 + 0.24 * dl
e = z - z28
sh = lambda v: float(v.std())
print(f'\nLGB-ансамбль: старый {float((z-zold).std()):.5f} · '
      f'новый {float((z-znew).std()):.5f} · Δ {float((z-zold).std())-float((z-znew).std()):+.5f}')
print(f'shape v23 {float((z-z_v23).std()):.5f} · shape v28 {sh(e):.5f} '
      f'(v28 лучше на {float((z-z_v23).std())-sh(e):+.5f})')

C = float(((e - e.mean()) * (d_full - d_full.mean())).mean())
D = float(((d_full - d_full.mean())**2).mean())
a = C / max(D, 1e-15); dM = C * C / max(D, 1e-15)
new = np.sqrt(max(e.var() - dM, 0))
print(f'\nd_depth поверх v28: std {d_full.std():.5f} · C {C:+.3e} · D {D:.3e}')
print(f'  alpha* {a:+.4f} · ΔM {dM:.3e} · shape {sh(e):.5f} -> {new:.5f} ({sh(e)-new:+.5f})')
print(f'  ПРИ alpha = 1: shape {float((e - (d_full - d_full.mean())).std()):.5f} '
      f'против {sh(e):.5f} '
      f'({sh(e) - float((e - (d_full - d_full.mean())).std()):+.5f})')
dann = np.load(f'/tmp/d_annual_{A}.npy'); d16 = np.load(O / f'dz_a{A}.npz')['dz']
print(f'\nдиагностика: corr(depth, life) {np.corrcoef(d_depth, d_life)[0,1]:+.4f} · '
      f'corr(depth, annual) {np.corrcoef(d_depth, dann)[0,1]:+.4f} · '
      f'corr(depth, GRU) {np.corrcoef(d_depth, d16)[0,1]:+.4f}')
np.savez_compressed(O / f'depth_geom_a{A}.npz', user_id=ref, d_depth=d_depth,
                    d_full=d_full, e28=e, z28=z28)
print('\nготово', flush=True)
