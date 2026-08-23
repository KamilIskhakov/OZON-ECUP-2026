"""v27_life = v23 + направление блока беcкэповой истории.

Направление уже содержит боевые веса семейств (0.4 LGB + 0.6 CatBoost)
внутри себя, поэтому коэффициент равен единице по построению: это
честная замена моделей на версию с новым блоком, а не подгонка.

Демеанирование обязательно: уровень v23 задан ручкой a_m = 0.18,
подобранной по лидерборду, и валидировалась форма, не уровень.

Аргумент выбирает компоненту:
  full (по умолчанию) 0.4 d_lgb + 0.6 d_cb — как в боевом рецепте
  lgb                 только LGB-часть
  cb                  только CatBoost-часть
"""
import sys, numpy as np, polars as pl; sys.path.insert(0, 'src')
from pathlib import Path

WHICH = sys.argv[1] if len(sys.argv) > 1 else 'full'
A = Path('artifacts')
F = np.load(A / 'neural' / 'longmoney_prod_a408.npz')
d_raw = {'full': F['d_tot'], 'lgb': 0.4 * F['d_lgb'], 'cb': 0.6 * F['d_cb']}[WHICH]
NAME = {'full': 'submission_v27_life.csv', 'lgb': 'submission_v27_life_lgb.csv',
        'cb': 'submission_v27_life_cb.csv'}[WHICH]

base = pl.read_csv(A / 'submission_v23_annual_opt.csv')
col = [c for c in base.columns if c != 'user_id'][0]
j = base.join(pl.DataFrame({'user_id': F['user_id'], 'd': d_raw}),
              on='user_id', how='left')
assert j['d'].null_count() == 0, 'не все пользователи покрыты направлением'

zb = np.log1p(j[col].to_numpy()); d = j['d'].to_numpy(); d = d - d.mean()
print(f'компонента {WHICH} · пользователей {len(zb):,}')
print(f'база: среднее z {zb.mean():.4f} · направление std {d.std():.5f}')
print(f'сдвиг уровня после демеанирования {d.mean():+.2e}')

out = base.with_columns(pl.Series(col, np.expm1(np.clip(zb + d, 0, None))))
v = out[col].to_numpy()
assert np.isfinite(v).all() and (v >= 0).all(), 'некорректные значения'
assert out['user_id'].to_list() == base['user_id'].to_list(), 'порядок строк разошёлся'
out.write_csv(A / NAME)
print(f'\nзаписан {NAME}')
print(f'  среднее {v.mean():.2f} (у v23 {base[col].to_numpy().mean():.2f})'
      f' · max {v.max():.0f} · нулей {(v < 1e-9).mean():.4f}')
ch = np.abs(np.log1p(v) - zb)
print(f'  сдвиг в z: медиана {np.median(ch):.5f} · p99 {np.quantile(ch, .99):.4f}')
