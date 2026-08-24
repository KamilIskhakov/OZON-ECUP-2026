"""Ортогональная цель: ошибка вне оболочки уже найденных направлений.

Попытка определить исторический «современный стек» через боевые
коэффициенты сломалась: на 258 и 288 такая база ХУЖЕ обычной OOF-пары
на 0.00286 и 0.00127, потому что локальные оптимумы направлений
отличаются от боевых.

Коэффициенты вообще не нужны. Берём честную стационарную базу z0_oof,
которая существует везде, и удаляем из ошибки всё, что линейно
объясняется уже известными направлениями:

    e_A    = z_A - z0_oof(A)
    D_A    = [d_GRU(A), d_ann(A), d_life(A)]   (центрированы)
    P_A    = D (D'D + lam I)^-1 D'
    r_A    = (I - P_A) e_A

Сеть учится предсказывать r_A. Постановка не содержит ни одного
подбираемого beta, поэтому межъякорная нестабильность коэффициентов
из неё исчезает.

Использование y_A при построении проекции законно: на обучающих
якорях y и есть метка, а коэффициенты проекции никуда не переносятся —
на каждом якоре своя.

ВАЖНО про утечку: подменяется ЦЕЛЬ (z -> z0 + r), а не z0. Иначе
метка попала бы в prior, который подаётся сети на вход и на боевом
якоре невычислим.
"""
import sys, warnings; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import load_panel
from strong_base import annual

O = Path('artifacts/neural'); LAM = 1e-6
df = load_panel()
cen = lambda v: v - np.nanmean(v)
print(f'{"якорь":>7}{"n":>10}{"std(e)":>10}{"std(r)":>10}{"q":>8}'
      f'{"gru":>8}{"ann":>8}{"life":>8}')
for A in (258, 288, 318, 348, 378):
    o = np.load(O / f'oofpm_a{A}.npz'); uid = o['user_id']
    key = pl.DataFrame({'user_id': uid})
    al = lambda u_, v_: key.join(
        pl.DataFrame({'user_id': u_, 'v': np.asarray(v_, dtype='float64')}),
        on='user_id', how='left')['v'].to_numpy()
    cand = [O / f'gru1_dz_a{A}.npz', O / f'gru1b_dz_a{A}.npz', O / f'gru1c_dz_a{A}.npz']
    g = np.load(next(c for c in cand if c.exists()))
    d_gru = np.nan_to_num(al(g['user_id'], g['dz']))
    d_ann = annual(df, A, uid)
    lf = O / f'life_dir_a{A}.npz'
    L = np.load(lf if lf.exists() else O / 'life_geom_a378.npz')
    d_life = np.nan_to_num(al(L['user_id'], L['d_life' if 'd_life' in L.files else 'd_life']))
    z = np.log1p(o['y']); z0 = o['z0'].astype('float64')
    e = z - z0
    D = np.column_stack([cen(d_gru), cen(d_ann), cen(d_life)])
    G = D.T @ D + LAM * np.eye(D.shape[1]) * np.trace(D.T @ D) / D.shape[1]
    b = np.linalg.solve(G, D.T @ cen(e))
    r = cen(e) - D @ b + e.mean()
    q = float(np.std(cen(e) - D @ b) / np.std(cen(e)))
    print(f'{A:>7}{len(uid):>10,}{e.std():>10.5f}{r.std():>10.5f}{q:>8.4f}'
          f'{b[0]:>8.3f}{b[1]:>8.4f}{b[2]:>8.3f}')
    out = {k: o[k] for k in o.files}
    out['z_override'] = (z0 + r).astype('float32')
    out['r_orth'] = r.astype('float32')
    np.savez_compressed(O / f'oofpm_orth_a{A}.npz', **out)
print('\nсохранены oofpm_orth_a{258,288,318,348,378}.npz')
print('q — доля стандартного отклонения ошибки, оставшаяся после проекции')
