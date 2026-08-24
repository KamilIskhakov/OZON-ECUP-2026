"""Маржинальный ресурс нового Gap-GRU поверх современного стека.

Протокол разный на двух якорях, потому что направления сохранены не везде.

ЯКОРЬ 378 — главный вопрос: осталась ли информация после всего стека.
Есть d_GRU1 (dz_a378), d_annual, d_life, поэтому

    e_28   = z - (z_backbone + 0.35 d_GRU1 + 0.0104 d_ann + 0.24 d_life)
    d_perp = d_C - P_[d_GRU1, d_ann, d_life] d_C
    C = E[e_28 d_perp],  V = E[d_perp^2],  alpha* = C/V

ЯКОРЬ 348 — вопрос о переносе архитектуры во времени. Полного стека
там нет и притворяться не буду: считается d_C - d_A против остатка
честного контроля.

Печатается corr(d_C, d_A): если новая сеть просто заново выучила
старое направление, ветка мертва независимо от величины выигрыша.
"""
import sys, json; sys.path.insert(0, 'src')
import numpy as np, polars as pl
from pathlib import Path

N = Path('artifacts/neural'); R = Path('artifacts/recon')
A_J = json.loads((N / 'gru_Aexact.json').read_text())
C_J = json.loads((N / 'gru_C.json').read_text())
print('=== сводка прогонов ===')
print(f'{"":<10}{"фолд 0":>22}{"фолд 1":>22}')
for nm, J in (('A_exact', A_J), ('C_long', C_J)):
    print(f'{nm:<10}' + ''.join(
        f'{f["gain"]:>+12.5f} (a {f["alpha"]:.2f})' for f in J))

dA = {f['fold']: np.asarray(f['dz'], dtype='float64') for f in A_J if 'dz' in f}
dC = {f['fold']: np.asarray(f['dz'], dtype='float64') for f in C_J if 'dz' in f}
if not dA or not dC:
    print('\nв json нет самих направлений — беру из чекпоинтов dz_*.npz')
    dA = {1: np.load(N / 'dzA_a378.npz')['dz']} if (N / 'dzA_a378.npz').exists() else {}
    dC = {1: np.load(N / 'dzC_a378.npz')['dz']} if (N / 'dzC_a378.npz').exists() else {}

G = np.load(N / 'life_geom_a378.npz'); ref = G['user_id']; d_life = G['d_life']
E = np.load(R / 'cb_ens2.npz'); dann = np.load(R / 'd_annual_378.npy')
d16 = np.load(N / 'dz_a378.npz')['dz']
o = np.load(N / 'oofpm_a378.npz')
key = pl.DataFrame({'user_id': ref})
zt = key.join(pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y'])}),
              on='user_id', how='left')['z'].to_numpy()
zl = np.mean([E[k] for k in E.files if k.startswith('lgb')], 0)
zc = np.mean([E[k] for k in E.files if k.startswith(('cb_', 'brd', 'dpw'))], 0)
bb = 0.4 * zl + 0.6 * zc
cen = lambda v: v - v.mean()
z28 = bb + 0.35 * cen(d16) + 0.0104 * cen(dann) + 0.24 * cen(d_life)
e28 = zt - z28
D = np.column_stack([cen(d16), cen(dann), cen(d_life)])
print(f'\n=== ЯКОРЬ 378 · shape v28-аналога {e28.std():.5f} ===')
for nm, d in (('A_exact', dA.get(1)), ('C_long', dC.get(1))):
    if d is None:
        print(f'  {nm}: направление недоступно'); continue
    d = cen(np.asarray(d, dtype='float64'))
    b = np.linalg.lstsq(D, d, rcond=None)[0]
    dp = d - D @ b
    for lab, v in (('сырое', d), ('ортогональное', dp)):
        C_ = float((cen(e28) * cen(v)).mean()); V_ = float((cen(v) ** 2).mean())
        a = C_ / max(V_, 1e-15); dM = C_ * C_ / max(V_, 1e-15)
        new = np.sqrt(max(e28.var() - dM, 0))
        print(f'  {nm} {lab:<14} std {v.std():.5f} · C {C_:+.3e} · alpha* {a:+.4f} · '
              f'shape {e28.std():.5f} -> {new:.5f} ({e28.std()-new:+.5f})')
    print(f'  {nm} ортогональная доля {dp.std()/d.std():.4f} · '
          f'corr с GRU1 {np.corrcoef(d, d16)[0,1]:+.4f} · '
          f'annual {np.corrcoef(d, dann)[0,1]:+.4f} · '
          f'life {np.corrcoef(d, d_life)[0,1]:+.4f}')
if dA.get(1) is not None and dC.get(1) is not None:
    print(f'\n  corr(d_C, d_A) = {np.corrcoef(dC[1], dA[1])[0,1]:+.4f}  '
          f'<- если высокая, C переоткрыл старое направление')
