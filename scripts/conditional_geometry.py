"""Условные коэффициенты найденных направлений: alpha(X), а не d(X).

Когортный аудит проверял E[e | G] и нашёл ноль: систематического
смещения по классам нет. Но это НЕ то же самое, что

    alpha_g = E[e d | G=g] / E[d^2 | G=g].

Может выполняться E[e|G] = 0 при alpha_1 = +0.5 и alpha_2 = -0.1:
когортного смещения нет, но одному типу пользователей направление
помогает сильно, другому почти нет. Глобальный alpha усредняет это и
теряет ресурс, которого линейная оболочка принципиально не видит.

Правдоподобия добавляет известное: годовой фильтр работает втрое
сильнее на 378, чем на ранних якорях, то есть его сила зависит от
контекста. Возможно, и от пользователя.

Протокол без обучения моделей:
  база — честный сильный стек на 348/378;
  направления — GRU, annual, life;
  гейты — расхождение LGB-CB, уровень базового прогноза, GMV30/GMV90;
  границы групп и локальные alpha берутся с 348, УСАЖИВАЮТСЯ к
  глобальному, и без изменений применяются к 378.

Добавка к прогнозу: (alpha_g - alpha_0) * d, то есть только
СВЕРХ глобального направления. Никакого oracle на 378.
"""
import sys, warnings; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import load_panel
from strong_base import annual

O = Path('artifacts/neural'); NG = 4
df = load_panel()
cen = lambda v: v - np.nanmean(v)


def load(A):
    s = np.load(O / f'oofpm_strong_a{A}.npz'); uid = s['user_id']
    z = np.log1p(s['y']); base = s['z0'].astype('float64')
    o = np.load(O / f'oofpm_a{A}.npz')
    key = pl.DataFrame({'user_id': uid})
    al = lambda u_, v_: key.join(
        pl.DataFrame({'user_id': u_, 'v': np.asarray(v_, 'float64')}),
        on='user_id', how='left')['v'].to_numpy()
    g = np.load(next(p for p in (O / f'gru1_dz_a{A}.npz', O / f'gru1b_dz_a{A}.npz',
                                 O / f'gru1c_dz_a{A}.npz') if p.exists()))
    lf = O / f'life_dir_a{A}.npz'
    L = np.load(lf if lf.exists() else O / f'life_geom_a{A}.npz')
    D = {'GRU': cen(np.nan_to_num(al(g['user_id'], g['dz']))),
         'annual': cen(annual(df, A, uid)),
         'life': cen(np.nan_to_num(al(L['user_id'], L['d_life'])))}
    w = df.filter(pl.col('d') <= A).group_by('user_id').agg(
        g30=(pl.col('gmv') * (pl.col('d') > A - 30)).sum(),
        g90=(pl.col('gmv') * (pl.col('d') > A - 90)).sum())
    j = key.join(w, on='user_id', how='left')
    g30 = j['g30'].fill_null(0.0).to_numpy(); g90 = j['g90'].fill_null(0.0).to_numpy()
    gates = {'расхождение LGB-CB': (o['z0_lgb'] - o['z0_cb']).astype('float64'),
             'уровень прогноза': base,
             'GMV30/GMV90': g30 / np.maximum(g90, 1e-9)}
    return uid, z - base, D, gates


A1, A2 = 348, 378
_, e1, D1, G1 = load(A1)
_, e2, D2, G2 = load(A2)
print(f'shape базы: {A1} {e1.std():.5f} · {A2} {e2.std():.5f}\n')
print(f'{"направление":<10}{"гейт":<22}{"a0":>8}{"локальные a_g":>34}'
      f'{"глоб.378":>11}{"услов.378":>11}{"Δ":>10}')
for dn in D1:
    d1, d2 = D1[dn], D2[dn]
    a0 = float((cen(e1) * d1).mean() / (d1 ** 2).mean())
    base2 = float((e2 - a0 * d2).std())
    g0 = float((cen(e2) * d2).mean() / (d2 ** 2).mean())
    print(f'{dn:<10}{"(глобальный)":<22}{a0:>8.3f}{"":>34}'
          f'{base2:>11.5f}{"":>11}{"":>10}  (опт.378 a={g0:.3f})')
    for gn in G1:
        q = np.quantile(G1[gn], np.linspace(0, 1, NG + 1)[1:-1])
        k1 = np.clip(np.digitize(G1[gn], q), 0, NG - 1)
        k2 = np.clip(np.digitize(G2[gn], q), 0, NG - 1)
        ag = np.zeros(NG)
        for j in range(NG):
            m = k1 == j
            if m.sum() < 100: ag[j] = a0; continue
            num = float((cen(e1)[m] * d1[m]).mean()); den = float((d1[m] ** 2).mean())
            n = int(m.sum()); lam = 20000.0     # усадка к глобальному
            ag[j] = (num * n + a0 * den * lam) / (den * n + den * lam)
        v = float((e2 - a0 * d2 - (ag[k2] - a0) * d2).std())
        print(f'{"":<10}{gn:<22}{"":>8}{" ".join(f"{x:+.3f}" for x in ag):>34}'
              f'{base2:>11.5f}{v:>11.5f}{base2-v:>+10.5f}')
print(f'\nусадка lam = 20000 · шлюз 3e-4 на 378')
