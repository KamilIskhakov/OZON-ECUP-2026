"""v26 = v23 + полная замена LGB-части (без ранней остановки + факторизация).

Направление уже содержит вес 0.4 LGB-части внутри себя, поэтому
коэффициент равен единице по построению и подгонки нет.

Демеанирование обязательно: уровень v23 задан ручкой a_m = 0.18,
подобранной по лидерборду, а валидировалась форма, не уровень.

Аргумент выбирает компоненту направления:
  full  (по умолчанию) 0.4 (z_alt - z_prod)   — полная замена
  noes                 0.4 (z_old - z_prod)   — только отказ от ранней остановки
  fact                 0.4 (z_alt - z_old)    — только факторизация (это и был v25)
"""
import sys, numpy as np, polars as pl; sys.path.insert(0, 'src')
from pathlib import Path

WHICH = sys.argv[1] if len(sys.argv) > 1 else 'full'
KEY = {'full': 'd_tot', 'noes': 'd_noes', 'fact': 'd_fact'}[WHICH]
NAME = {'full': 'submission_v26_noes_fact_lgb.csv',
        'noes': 'submission_v26b_noes_only.csv',
        'fact': 'submission_v26c_fact_only.csv'}[WHICH]

A = Path('artifacts')
base = pl.read_csv(A / 'submission_v23_annual_opt.csv')
col = [c for c in base.columns if c != 'user_id'][0]
F = np.load(A / 'neural' / 'altlgb_prod_a408.npz')
j = base.join(pl.DataFrame({'user_id': F['user_id'], 'd': F[KEY]}),
              on='user_id', how='left')
assert j['d'].null_count() == 0, 'не все пользователи покрыты направлением'

zb = np.log1p(j[col].to_numpy()); d = j['d'].to_numpy(); d = d - d.mean()
print(f'компонента {WHICH} ({KEY}) · пользователей {len(zb):,}')
print(f'база: среднее z {zb.mean():.4f} · направление std {d.std():.5f}')
print(f'сдвиг уровня после демеанирования {d.mean():+.2e}')

out = base.with_columns(pl.Series(col, np.expm1(np.clip(zb + d, 0, None))))
v = out[col].to_numpy()
assert np.isfinite(v).all() and (v >= 0).all(), 'некорректные значения'
sample = pl.read_csv('data/sample_submit.csv') if Path('data/sample_submit.csv').exists() else None
if sample is not None:
    assert out['user_id'].to_list() == sample['user_id'].to_list(), 'порядок строк разошёлся'
out.write_csv(A / NAME)
print(f'\nзаписан {NAME}')
print(f'  среднее {v.mean():.2f} (у v23 {base[col].to_numpy().mean():.2f})'
      f' · max {v.max():.0f} · нулей {(v < 1e-9).mean():.4f}')
ch = np.abs(np.log1p(v) - zb)
print(f'  сдвиг в z: медиана {np.median(ch):.5f} · p99 {np.quantile(ch, .99):.4f}')
