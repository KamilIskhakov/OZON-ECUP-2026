"""Аудит готовых сабмитов: разнообразие и выбор финальной пары.

Ни одной новой модели. Две части.

ЧАСТЬ 1 — расхождение прогнозов на боевом якоре 408, все шесть
кандидатов. Не требует разметки.

ЧАСТЬ 2 — корреляция ОШИБОК и бутстрап public/private на якоре 378,
где разметка есть. Реконструируются только те версии, которые
восстанавливаются точно или почти точно:

  v23 = 0.4 zl_saved + 0.6 zc + 0.35 (d16 - mean) + 0.0104 (dann - mean)
  v28 = v23 + 0.24 (d_life - mean)
  v25 = v23 + 0.4 (zl_ord - zl_old), демеанировано
  v22 = v23 с годовым коэффициентом, делённым на 1.159 (вершина
        параболы, по которой v23 и получен из v22)

v18 и v20 предшествуют годовому фильтру, их направления на 378 не
сохранены, поэтому они участвуют только в части 1.

Реконструкция v22 опирается на допущение, что между v22 и v23
изменился ТОЛЬКО коэффициент годового направления. Оно проверяется:
величина расхождения v23-v22 на 408 сравнивается с предсказанной.
"""
import sys, numpy as np, polars as pl; sys.path.insert(0, 'src')
from pathlib import Path

A = Path('artifacts'); N = A / 'neural'; R = A / 'recon'
PUB = {'v18': None, 'v20': None, 'v22': 1.6479970, 'v23': 1.6479820,
       'v25': 1.6480404, 'v28': 1.6478363}
F = {'v18': 'submission_v18_2dir.csv', 'v20': 'submission_v20_yoy_scaled.csv',
     'v22': 'submission_v22_annual.csv', 'v23': 'submission_v23_annual_opt.csv',
     'v25': 'submission_v25_facthead.csv', 'v28': 'submission_v28_life_a024.csv'}

# ---------- часть 1: расхождение на 408
Z = {}
for k, f in F.items():
    d = pl.read_csv(A / f)
    c = [x for x in d.columns if x != 'user_id'][0]
    Z[k] = np.log1p(d[c].to_numpy())
ks = list(F)
print('=== ЧАСТЬ 1 · расхождение прогнозов на боевом якоре 408 ===')
print(f'{"пара":<12}{"std(d)":>10}{"E|d|":>10}{"q99|d|":>10}{"corr(z)":>11}')
for i, a in enumerate(ks):
    for b in ks[i+1:]:
        d = Z[a] - Z[b]
        print(f'{a+"~"+b:<12}{d.std():>10.5f}{np.abs(d).mean():>10.5f}'
              f'{np.quantile(np.abs(d),.99):>10.4f}{np.corrcoef(Z[a],Z[b])[0,1]:>11.6f}')

# ---------- часть 2: реконструкция на 378
G = np.load(N / 'life_geom_a378.npz')
ref = G['user_id']; z23 = G['z_v23']; d_life = G['d_life']
E = np.load(R / 'cb_ens2.npz'); dann = np.load(R / 'd_annual_378.npy')
FE = np.load(N / 'fact_ensemble.npz')
o = np.load(N / 'oofpm_a378.npz')
key = pl.DataFrame({'user_id': ref})
al = lambda u, v: key.join(pl.DataFrame({'user_id': u, 'v': v}),
                           on='user_id', how='left')['v'].to_numpy()
zt = al(o['user_id'], np.log1p(o['y']))
d_fact = al(FE['user_id'], 0.4 * (FE['zl_ord'] - FE['zl_old']))
V = {'v23': z23,
     'v28': z23 + 0.24 * (d_life - d_life.mean()),
     'v25': z23 + (d_fact - d_fact.mean()),
     'v22': z23 - 0.0104 * (1 - 1 / 1.159) * (dann - dann.mean())}
print(f'\n=== ЧАСТЬ 2 · якорь 378, {len(ref):,} пользователей ===')
pred_v22 = 0.0104 * (1 - 1 / 1.159) * dann.std()
real_v22 = (Z['v23'] - Z['v22']).std()
print(f'проверка реконструкции v22: предсказанный std расхождения '
      f'{pred_v22:.5f} против фактического на 408 {real_v22:.5f} '
      f'(отношение {pred_v22/max(real_v22,1e-12):.2f})')
er = {k: zt - v for k, v in V.items()}
kk = ['v22', 'v23', 'v25', 'v28']
print(f'\n{"кандидат":<10}{"shape 378":>12}{"против v28":>13}')
for k in kk:
    print(f'{k:<10}{er[k].std():>12.5f}{er["v28"].std()-er[k].std():>+13.5f}')
print(f'\nкорреляция ОШИБОК:')
print(f'{"":<10}' + ''.join(f'{k:>10}' for k in kk))
for a in kk:
    print(f'{a:<10}' + ''.join(f'{np.corrcoef(er[a],er[b])[0,1]:>10.5f}' for b in kk))

# ---------- бутстрап 20/80
rng = np.random.default_rng(0); B = 1000
E2 = np.column_stack([er[k]**2 for k in kk])
n = len(zt); pub = np.zeros((B, len(kk))); prv = np.zeros((B, len(kk)))
for b in range(B):
    m = rng.random(n) < 0.20
    pub[b] = np.sqrt(E2[m].mean(0)); prv[b] = np.sqrt(E2[~m].mean(0))
i28 = kk.index('v28')
print(f'\n=== БУТСТРАП 20/80, {B} итераций ===')
print(f'{"кандидат":<10}{"P(лучше v28":>14}{"E[разница":>13}{"std разницы":>13}')
print(f'{"":<10}{"на private)":>14}{"на private]":>13}{"":>13}')
for j, k in enumerate(kk):
    d = prv[:, j] - prv[:, i28]
    print(f'{k:<10}{(d<0).mean():>14.3f}{-d.mean():>+13.6f}{d.std():>13.6f}')
am_p = pub.argmin(1); am_v = prv.argmin(1)
print(f'\nP(лучший по public = лучший по private) = {(am_p==am_v).mean():.3f}')
print(f'частота «лучший по public»:  ' +
      ' '.join(f'{kk[j]} {(am_p==j).mean():.3f}' for j in range(len(kk))))
print(f'частота «лучший по private»: ' +
      ' '.join(f'{kk[j]} {(am_v==j).mean():.3f}' for j in range(len(kk))))
sel = prv[np.arange(B), am_p]
print(f'\n=== цена выбора по публичной части ===')
print(f'  выбор лучшего по 20 %:  private {sel.mean():.6f}')
print(f'  всегда v28:             private {prv[:, i28].mean():.6f}')
print(f'  оракул (лучший из всех): private {prv.min(1).mean():.6f}')
print(f'  сожаление выбора по public: {sel.mean()-prv.min(1).mean():+.6f}')
print(f'  сожаление «всегда v28»:     {prv[:,i28].mean()-prv.min(1).mean():+.6f}')
best2 = {}
for i in range(len(kk)):
    for j in range(i+1, len(kk)):
        best2[(kk[i], kk[j])] = np.minimum(prv[:, i], prv[:, j]).mean()
print(f'\n=== лучшая ПАРА финальных кандидатов (min по двум на private) ===')
for p, v in sorted(best2.items(), key=lambda x: x[1])[:6]:
    print(f'  {p[0]} + {p[1]:<6} {v:.6f}')
