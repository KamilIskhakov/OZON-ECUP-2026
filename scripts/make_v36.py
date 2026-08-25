"""v36 = v34 + BTYD-направление.

Коэффициент не подбирается: берётся по правилу применённой амплитуды.
На валидации оптимум alpha* = 0.443 при std ортогонального 0.0334,
то есть амплитуда 0.0148. Перенос применённой амплитуды на боевой
якорь по прецеденту нейронаправления составляет 0.73, отсюда целевая
боевая амплитуда 0.0108.
"""
import sys, numpy as np, polars as pl; sys.path.insert(0, 'src')
from pathlib import Path
A = Path('artifacts'); TARGET = 0.0108
F = np.load(A / 'neural' / 'btyd_prod_a408.npz')
b = pl.read_csv(A / 'submission_v34_gru094.csv')
col = [c for c in b.columns if c != 'user_id'][0]
j = b.join(pl.DataFrame({'user_id': F['user_id'], 'd': F['d_orth']}),
           on='user_id', how='left')
assert j['d'].null_count() == 0, 'не все пользователи покрыты'
zb = np.log1p(j[col].to_numpy()); d = j['d'].to_numpy(); d = d - d.mean()
ALPHA = TARGET / d.std()
out = b.with_columns(pl.Series(col, np.expm1(np.clip(zb + ALPHA * d, 0, None))))
v = out[col].to_numpy()
assert np.isfinite(v).all() and (v >= 0).all()
assert out['user_id'].to_list() == b['user_id'].to_list()
out.write_csv(A / 'submission_v36_btyd.csv')
print(f'записан submission_v36_btyd.csv · alpha {ALPHA:.4f}')
print(f'  направление std {d.std():.5f} · применено {(ALPHA*d).std():.5f}')
print(f'  среднее {v.mean():.2f} (v34 {b[col].to_numpy().mean():.2f}) · '
      f'нулей {(v < 1e-9).mean():.4f}')
print(f'  сдвиг в z: медиана {np.median(np.abs(ALPHA*d)):.5f} · '
      f'p99 {np.quantile(np.abs(ALPHA*d), .99):.4f}')
print(f'  текущий v34 = 1.6473878 · ожидание около 1.647317')
