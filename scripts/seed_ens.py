"""Таблица C, D, G по сидам и трёхсидовому ансамблю на якоре 378.

Усреднять сырые направления нельзя: их масштаб различается. Каждое
нормируется на свой std, затем равные веса без подгонки.

Решающее сравнение — G ансамбля против G лучшего одиночного сида.
При rho = 0.87 и K = 3 модель независимого шума предсказывает рост
примерно на 9.5 %. Существенно больший рост означал бы, что сиды
находят РАЗНЫЕ полезные компоненты, а не только шум.
"""
import sys; sys.path.insert(0, 'src')
import numpy as np, polars as pl
from pathlib import Path

N = Path('artifacts/neural'); R = Path('artifacts/recon')
G = np.load(N / 'life_geom_a378.npz'); ref = G['user_id']; d_life = G['d_life']
E = np.load(R / 'cb_ens2.npz'); dann = np.load(R / 'd_annual_378.npy')
d16 = np.load(N / 'dz_a378.npz')['dz']
o = np.load(N / 'oofpm_a378.npz')
key = pl.DataFrame({'user_id': ref})
al = lambda u, v: key.join(
    pl.DataFrame({'user_id': u, 'v': np.asarray(v, dtype='float64')}),
    on='user_id', how='left')['v'].to_numpy()
zt = al(o['user_id'], np.log1p(o['y']))
zl = np.mean([E[k] for k in E.files if k.startswith('lgb')], 0)
zc = np.mean([E[k] for k in E.files if k.startswith(('cb_', 'brd', 'dpw'))], 0)
cen = lambda v: v - v.mean()
z28 = 0.4 * zl + 0.6 * zc + 0.35 * cen(d16) + 0.0104 * cen(dann) + 0.24 * cen(d_life)
e = zt - z28
D3 = np.column_stack([cen(d16), cen(dann), cen(d_life)])
sh0 = float(e.std())
print(f'shape v28-аналога {sh0:.5f}\n')

FILES = {42: N / 'gru4_matched_dz_a378.npz',
         7: N / 'gru_val_s7_dz_a378.npz',
         2026: N / 'gru_val_s2026_dz_a378.npz'}
raw, nrm = {}, {}
for s, f in FILES.items():
    if not f.exists():
        print(f'сид {s}: файла нет ({f.name})'); continue
    k = np.load(f)
    d = cen(np.nan_to_num(al(k['user_id'], k['dz'])))
    raw[s] = d; nrm[s] = d / d.std()

def stat(d, lab):
    b = np.linalg.lstsq(D3, d, rcond=None)[0]
    dp = d - D3 @ b
    C = float((cen(e) * cen(dp)).mean()); V = float((cen(dp) ** 2).mean())
    g = C * C / max(V, 1e-15)
    new = np.sqrt(max(e.var() - g, 0))
    print(f'{lab:<14}std {d.std():.5f} · орт.доля {dp.std()/d.std():.3f} · '
          f'C {C:+.3e} · alpha* {C/max(V,1e-15):+.4f} · '
          f'G {g:.3e} · прирост {sh0-new:+.6f}')
    return g, sh0 - new

res = {}
for s in sorted(raw):
    res[s] = stat(raw[s], f'сид {s}')
if len(nrm) >= 2:
    ens = np.mean([nrm[s] for s in sorted(nrm)], 0)
    print()
    ge, de = stat(ens, f'ансамбль {len(nrm)}')
    best = max(res.values(), key=lambda r: r[1])
    print(f'\nлучший одиночный прирост {best[1]:+.6f} · ансамбль {de:+.6f} · '
          f'разница {de-best[1]:+.6f}')
    print(f'рост G: {ge/max(best[0],1e-15)-1:+.1%} (модель шума предсказывает +9.5 %)')
    print(f'порог отправки: разница >= 1e-5')
    print(f'\nкорреляции направлений:')
    ks = sorted(nrm)
    print('       ' + ''.join(f'{k:>10}' for k in ks))
    for a in ks:
        print(f'{a:<7}' + ''.join(f'{np.corrcoef(nrm[a],nrm[b])[0,1]:>10.4f}' for b in ks))
