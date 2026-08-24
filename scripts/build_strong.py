"""Сборка честной сильной базы на 318/348/378 и её проверка.

    z0_strong(A) = z0_oof(A) + 0.35 gru1(A) + 0.0104 ann(A) + 0.24 life(A)

Все компоненты честны относительно A. Коэффициенты боевые, НЕ
переоптимизируются на каждом якоре.

Проверка: сильная база обязана быть лучше z0_oof на каждом якоре, и на
378 — сопоставима с реальным v28-аналогом. Если нет, дальше идти нельзя.

Результат пишется в oofpm_strong_a{A}.npz с той же схемой полей, что и
oofpm, чтобы train_gapgru.py принял его без правок: подменяется только
z0, остальное копируется.
"""
import sys, warnings; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import load_panel
sys.path.insert(0, 'scripts')
from strong_base import annual

O = Path('artifacts/neural'); R = Path('artifacts/recon')
W_GRU, W_ANN, W_LIFE = 0.35, 0.0104, 0.24
df = load_panel()
cen = lambda v: v - np.nanmean(v)

for A in (318, 348, 378):
    o = np.load(O / f'oofpm_a{A}.npz'); uid = o['user_id']
    key = pl.DataFrame({'user_id': uid})
    al = lambda u_, v_: key.join(
        pl.DataFrame({'user_id': u_, 'v': np.asarray(v_, dtype='float64')}),
        on='user_id', how='left')['v'].to_numpy()
    g = np.load(O / f'gru1_dz_a{A}.npz')
    d_gru = np.nan_to_num(al(g['user_id'], g['dz']))
    d_ann = annual(df, A, uid)
    lf = O / f'life_dir_a{A}.npz'
    if lf.exists():
        L = np.load(lf); d_life = np.nan_to_num(al(L['user_id'], L['d_life']))
    else:
        G = np.load(O / 'life_geom_a378.npz')
        d_life = np.nan_to_num(al(G['user_id'], G['d_life']))
    z0 = o['z0'].astype('float64'); z = np.log1p(o['y'])
    strong = z0 + W_GRU * cen(d_gru) + W_ANN * cen(d_ann) + W_LIFE * cen(d_life)
    s0, s1 = float((z - z0).std()), float((z - strong).std())
    print(f'якорь {A}: n {len(uid):,} · z0_oof {s0:.5f} -> сильная {s1:.5f} '
          f'({s0-s1:+.5f})', flush=True)
    print(f'   вклады: gru std {(W_GRU*cen(d_gru)).std():.5f} · '
          f'ann {(W_ANN*cen(d_ann)).std():.5f} · life {(W_LIFE*cen(d_life)).std():.5f}',
          flush=True)
    out = {k: o[k] for k in o.files}
    out['z0'] = strong.astype('float32')
    out['z0_oof_orig'] = z0.astype('float32')
    np.savez_compressed(O / f'oofpm_strong_a{A}.npz', **out)
print('\nсохранены oofpm_strong_a{318,348,378}.npz', flush=True)
