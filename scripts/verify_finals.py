"""Проверка воспроизводимости финальных сабмитов из сохранённых артефактов.

Каждый шаг цепочки пересобирается заново и сверяется с записанным
файлом. Если расхождение больше 1e-9 в z-шкале, значит артефакт или
коэффициент утрачен и финал не восстановим.

    v34 = v32 + 0.0943 * gruprod_orth
    v36 = v34 + 0.2860 * btyd_orth
    v37 = v36 + hull2d по осям [v34, v27]
    hedge = v23 + (1/3)(v37 - v23)
"""
import numpy as np, polars as pl
from pathlib import Path

A = Path('artifacts'); N = A / 'neural'
col = None


def load(name):
    global col
    d = pl.read_csv(A / name)
    if col is None:
        col = [c for c in d.columns if c != 'user_id'][0]
    return d['user_id'].to_numpy(), np.log1p(d[col].to_numpy())


def direc(npz, uid, key='d_orth'):
    f = np.load(N / npz)
    m = pl.DataFrame({'user_id': uid}).join(
        pl.DataFrame({'user_id': f['user_id'], 'v': f[key].astype('float64')}),
        on='user_id', how='left')['v'].to_numpy()
    assert np.isfinite(m).all(), f'{npz}: не все пользователи покрыты'
    return m - m.mean()


def check(lab, got, ref):
    e = float(np.abs(got - ref).max())
    print(f'  {lab:<28}max|Δz| {e:.2e}  {"ok" if e < 1e-9 else "РАСХОЖДЕНИЕ"}')
    return e < 1e-9


ok = True
u32, z32 = load('submission_v32_span20.csv')
u34, z34 = load('submission_v34_gru094.csv')
u36, z36 = load('submission_v36_btyd.csv')
u37, z37 = load('submission_v37_hull2d.csv')
u23, z23 = load('submission_v23_annual_opt.csv')
u27, z27 = load('submission_v27_life.csv')
uh, zh = load('submission_final_hedge33.csv')
for nm, u in (('v34', u34), ('v36', u36), ('v37', u37), ('v23', u23),
              ('v27', u27), ('hedge', uh)):
    assert np.array_equal(u32, u), f'{nm}: порядок строк расходится с v32'
print('порядок строк совпадает во всех файлах · строк', f'{len(u32):,}\n')

ok &= check('v32 + gru(0.0943) = v34', z32 + 0.0943 * direc('gruprod_dir_a408.npz', u32), z34)
# Коэффициенты пересчитываются ТОЧНО, как при сборке: округлённые
# значения из логов дают расхождение 3e-5 и маскируют настоящую проверку.
d_bt = direc('btyd_prod_a408.npz', u32)
a_bt = 0.0108 / d_bt.std()          # целевая применённая амплитуда
ok &= check(f'v34 + btyd({a_bt:.5f}) = v36', z34 + a_bt * d_bt, z36)

# Двумерная оболочка: точное решение G a = c, а не округлённые веса.
PUB = {'v27': 1.6492598146, 'v34': 1.6473877679, 'v36': 1.647288635}
D = np.column_stack([z34 - z36, z27 - z36])
G = D.T @ D / len(D)
c = np.array([(PUB['v36'] ** 2 - PUB[k] ** 2 + G[i, i]) / 2
              for i, k in enumerate(('v34', 'v27'))])
al = np.linalg.solve(G, c)
ok &= check(f'v36 + hull2d({al[0]:+.4f},{al[1]:+.4f}) = v37', z36 + D @ al, z37)
ok &= check('v23 + (1/3)(v37-v23) = hedge', z23 + (1 / 3) * (z37 - z23), zh)

print(f'\nитог: {"ВСЕ ШАГИ ВОСПРОИЗВОДЯТСЯ" if ok else "ЕСТЬ РАСХОЖДЕНИЯ"}')
for f in ('submission_v37_hull2d.csv', 'submission_final_hedge33.csv'):
    d = pl.read_csv(A / f); v = d[col].to_numpy()
    print(f'  {f}: строк {len(v):,} · конечных {np.isfinite(v).all()} · '
          f'неотрицательных {(v >= 0).all()} · среднее {v.mean():.2f}')
