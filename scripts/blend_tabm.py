"""Смешивание v23 с боевым направлением TabM и цена измерения.

Направление демеанируется, как d16 и dann в v22: у TabM среднее сильно
ниже базы, и сырое подмешивание сдвинуло бы уровень на -0.13, то есть
измеряло бы уровень вместо формы.
"""
import sys, numpy as np, polars as pl; sys.path.insert(0,'src')
from pathlib import Path
A = Path('artifacts')
base = pl.read_csv(A/'submission_v23_annual_opt.csv')
col = [c for c in base.columns if c != 'user_id'][0]
T = np.load(A/'neural/tabm_prod_v1_a408.npz')
m = pl.DataFrame({'user_id': T['uid'], 'zt': T['z']})
j = base.join(m, on='user_id', how='left')
assert j['zt'].null_count() == 0, 'не все пользователи покрыты'
zb = np.log1p(j[col].to_numpy()); zt = j['zt'].to_numpy()
d = zt - zb; d -= d.mean()
D = float((d**2).mean()); mse0 = 1.6479820**2
print(f'пользователей {len(zb):,} · база: среднее z {zb.mean():.4f}')
print(f'TabM: среднее z {zt.mean():.4f} · доля нулей {(zt<1e-9).mean():.4f}')
print(f'D = E[d²] = {D:.6f} · std(d) = {np.sqrt(D):.4f}\n')
print(f'{"lam":>7}{"цена при C=0":>14}{"выигрыш при C по валид.":>26}')
C_val = 0.0914 * D          # alpha = C/D, с фолда 378
for lam in (0.05, 0.07, 0.085, 0.09, 0.10):
    worst = np.sqrt(mse0 + lam*lam*D) - np.sqrt(mse0)
    best = np.sqrt(mse0) - np.sqrt(mse0 - 2*lam*C_val + lam*lam*D)
    print(f'{lam:>7.3f}{worst:>+14.5f}{best:>+26.5f}')
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.09
out = base.with_columns(pl.Series(col, np.expm1(np.clip(zb + LAM*d, 0, None))))
name = f'submission_v24_tabm{str(LAM).replace(".","")}.csv'
out.write_csv(A/name); print(f'\nзаписан {name} при lam = {LAM}')
