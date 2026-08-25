"""Когортный аудит остатка: где стек систематически смещён.

Вопрос не «какая модель лучше предсказывает каждого», а «для каких
КЛАССОВ пользователей весь стек смещён одинаково». Деревья, GRU,
годовой фильтр и линейная оболочка могли унаследовать одно смещение.

База — честная сильная реконструкция на каждом якоре:
    z_base(A) = z0_oof(A) + 0.35 gru1 + 0.0104 ann + 0.24 life

Для каждого скалярного признака, доступного и на 408, делаем четыре
равные группы и считаем mu_j = E[e | G_j].

Главное — НЕ oracle на одном якоре, а ПЕРЕНОС: границы групп и mu_j
берутся с 348 и применяются к 378. Oracle печатается только как
верхняя граница.

Разрыв до первого места 0.00233 требует направления с rho(e,d) ~ 0.053.
Одна четырёхгрупповая разбивка с weighted RMS смещений около 0.09
содержала бы его целиком.
"""
import sys, warnings; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import load_panel

O = Path('artifacts/neural'); NG = 4
df = load_panel()
cen = lambda v: v - np.nanmean(v)


def build(A):
    s = np.load(O / f'oofpm_strong_a{A}.npz')
    uid = s['user_id']; z = np.log1p(s['y']); base = s['z0'].astype('float64')
    e = z - base
    o = np.load(O / f'oofpm_a{A}.npz')
    key = pl.DataFrame({'user_id': uid})
    al = lambda u_, v_: key.join(
        pl.DataFrame({'user_id': u_, 'v': np.asarray(v_, dtype='float64')}),
        on='user_id', how='left')['v'].to_numpy()
    g = np.load(next(p for p in (O / f'gru1_dz_a{A}.npz', O / f'gru1b_dz_a{A}.npz',
                                 O / f'gru1c_dz_a{A}.npz') if p.exists()))
    L = np.load(O / f'life_dir_a{A}.npz')
    w = df.filter(pl.col('d') <= A)
    agg = (w.group_by('user_id').agg(
        g30=(pl.col('gmv') * (pl.col('d') > A - 30)).sum(),
        g90=(pl.col('gmv') * (pl.col('d') > A - 90)).sum(),
        o30=(pl.col('to_ord') * (pl.col('d') > A - 30)).sum(),
        bd90=((pl.col('gmv') > 0) & (pl.col('d') > A - 90)).sum(),
        srch=(pl.col('searches') * (pl.col('d') > A - 90)).sum(),
        gs=(pl.col('gmv_search') * (pl.col('d') > A - 90)).sum(),
        last=pl.col('d').filter(pl.col('gmv') > 0).max(),
        first=pl.col('d').min(), nact=pl.len()))
    j = key.join(agg, on='user_id', how='left')
    num = lambda c, f=0.0: j[c].fill_null(f).to_numpy().astype('float64')
    g30, g90, o30 = num('g30'), num('g90'), num('o30')
    bd90, srch, gs = num('bd90'), num('srch'), num('gs')
    last = j['last'].to_numpy().astype('float64')
    first = num('first'); nact = num('nact')
    feats = {
        'уровень прогноза': base,
        'p_buy': o['p0'].astype('float64'),
        'm_сумма': o['m0'].astype('float64'),
        'расхождение LGB-CB': (o['z0_lgb'] - o['z0_cb']).astype('float64'),
        'направление GRU': al(g['user_id'], g['dz']),
        'направление annual': al(uid, np.zeros(len(uid))),  # заполнится ниже
        'направление life': al(L['user_id'], L['d_life']),
        'GMV 30': np.log1p(g30), 'GMV 90': np.log1p(g90),
        'заказов 30': o30, 'покупочных дней 90': bd90,
        'поисков 90': np.log1p(srch),
        'доля Search': gs / np.maximum(g90, 1e-9),
        'recency покупки': np.where(np.isfinite(last), A - last, 999.0),
        'длина истории': A - first, 'активных дней': nact,
        'GMV30 / GMV90': g30 / np.maximum(g90, 1e-9),
    }
    from strong_base import annual
    feats['направление annual'] = annual(df, A, uid)
    return uid, e, {k: np.nan_to_num(v) for k, v in feats.items()}


U, E, F = {}, {}, {}
for A in (348, 378):
    U[A], E[A], F[A] = build(A)
    print(f'якорь {A}: n {len(U[A]):,} · shape базы {E[A].std():.5f} · '
          f'смещение {E[A].mean():+.5f}', flush=True)

R = float(E[378].std())
print(f'\nразрыв до первого места требует rho(e,d) ~ 0.053\n')
print(f'{"признак":<22}{"oracle 348":>12}{"oracle 378":>12}{"ПЕРЕНОС":>12}'
      f'{"RMS смещ.":>11}')
rows = []
for k in F[348]:
    x1, x2 = F[348][k], F[378][k]
    q = np.quantile(x1, np.linspace(0, 1, NG + 1)[1:-1])
    g1 = np.clip(np.digitize(x1, q), 0, NG - 1)
    g2 = np.clip(np.digitize(x2, q), 0, NG - 1)
    mu1 = np.array([E[348][g1 == j].mean() if (g1 == j).sum() else 0.0
                    for j in range(NG)])
    mu2 = np.array([E[378][g2 == j].mean() if (g2 == j).sum() else 0.0
                    for j in range(NG)])
    p1 = np.array([(g1 == j).mean() for j in range(NG)])
    p2 = np.array([(g2 == j).mean() for j in range(NG)])
    orc1 = E[348].std() - np.sqrt(max(E[348].var() - (p1 * mu1 ** 2).sum()
                                      + E[348].mean() ** 2 * 0, 0))
    o1 = float(E[348].std()); o2 = float(E[378].std())
    orc1 = o1 - float((E[348] - mu1[g1]).std())
    orc2 = o2 - float((E[378] - mu2[g2]).std())
    tr = o2 - float((E[378] - mu1[g2]).std())
    rms = float(np.sqrt((p2 * mu2 ** 2).sum()))
    rows.append((k, orc1, orc2, tr, rms))
for k, a1, a2, tr, rms in sorted(rows, key=lambda r: -r[3]):
    print(f'{k:<22}{a1:>+12.5f}{a2:>+12.5f}{tr:>+12.5f}{rms:>11.4f}')
print(f'\nПЕРЕНОС — mu с 348 применены к 378 по границам 348. Это и есть'
      f' то, что переносимо.')
