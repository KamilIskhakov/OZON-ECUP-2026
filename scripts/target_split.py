"""Две переparameterизации таргета: по горизонту и по каналу.

  A   контроль          одна голова 600 на log(1+Y)
  H   горизонт          3 x 200 на Y[1:10], Y[11:20], Y[21:30]
  S   канал             2 x 300 на Y_search и Y_cat

Подцелевые прогнозы НЕ суммируются в исходной шкале: expm1 от
E[log(1+Y_k)|X] даёт величину порядка условной медианы, а не среднего,
и при 40 % нулей в декаде разрыв Йенсена огромен — первая версия
этого эксперимента померила именно его (-0.014 при t = -171), а не
гипотезу. Вместо этого подцелевые прогнозы служат входами линейного
комбинатора, обученного на тех же обучающих якорях:

    m = alpha_0 + sum_k alpha_k * p_k

Он сам находит нужный масштаб в лог-шкале, поэтому вопрос Йенсена
снимается тождественно. Четыре параметра на сотни тысяч строк —
переобучение комбинатора пренебрежимо.

Механизм для горизонта: связь недавнего поведения с GMV через три дня
и через двадцать семь почти наверняка разная, а сейчас одна голова
обязана выучить обе. Для канала: gmv = gmv_search + gmv_cat выполняется
точно, но каналы очень разные — search присутствует в 80.7 % строк,
cat в 15.6 %, оба сразу лишь в 11.5 %.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
import lightgbm as lgb

ANCH = (348, 378); O = Path('artifacts/neural'); SEEDS = (42, 7, 2026); BUDGET = 600
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
HOR = ((1, 10), (11, 20), (21, 30))


def parts(a):
    """GMV в подокнах горизонта и по каналам за (a, a+30]."""
    w = df.filter(pl.col('d').is_between(a + 1, a + 30))
    agg = [ (pl.col('gmv') * pl.col('d').is_between(a + lo, a + hi)).sum()
            .alias(f'h{lo}') for lo, hi in HOR]
    agg += [pl.col('gmv_search').sum().alias('gs'), pl.col('gmv_cat').sum().alias('gc')]
    return w.group_by('user_id').agg(agg)


def reg(X, y, w, n, s):
    p = dict(ModelConfig(seed=s).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1); p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(random_state=s, **p).fit(X, y, sample_weight=w)


for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    P = (pl.concat([key.filter(pl.col('_a') == a).join(parts(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row').fill_null(0.0))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = y > 0
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Xp, wp = X[pos], w[pos]; zp = np.log1p(y[pos]) - zo[pos]
    cols_h = [f'h{lo}' for lo, _ in HOR]; cols_s = ['gs', 'gc']
    sub = {c: np.log1p(P[c].to_numpy()[pos]) for c in cols_h + cols_s}
    for c in cols_h + cols_s:
        print(f'  {c}: доля нулей среди покупателей {(sub[c] < 1e-9).mean():.3f}', flush=True)
    print(f'=== якорь {A} · обучающие {an} ===', flush=True)

    def combine(cols, cap, s):
        tr, va = [], []
        for c in cols:
            m = reg(Xp, sub[c], wp, cap, s)
            tr.append(m.predict(Xp)); va.append(m.predict(Xva))
        Atr = np.column_stack([np.ones(len(zp))] + tr)
        Ava = np.column_stack([np.ones(len(Xva))] + va)
        G = (Atr * wp[:, None]).T @ Atr
        al = np.linalg.solve(G, (Atr * wp[:, None]).T @ np.log1p(y[pos]))
        return Ava @ al, al

    R = {}
    for s in SEEDS:
        t0 = time.perf_counter()
        R[f'A_{s}'] = reg(Xp, zp, wp, BUDGET, s).predict(Xva) + last.l_plus
        R[f'H_{s}'], ah = combine(cols_h, BUDGET // 3, s)
        R[f'S_{s}'], as_ = combine(cols_s, BUDGET // 2, s)
        # аддитивная склейка с поправкой масштаба: Y = Y_s + Y_c верно в
        # ИСХОДНОЙ шкале, поэтому лог-линейный комбинатор для канала
        # неверно специфицирован. Поправка c_k снимает смещение Йенсена,
        # оставляя правильную функциональную форму.
        for nm, cols, cap in (('HA', cols_h, BUDGET // 3), ('SA', cols_s, BUDGET // 2)):
            acc = np.zeros(len(Xva))
            for c in cols:
                m = reg(Xp, sub[c], wp, cap, s)
                ptr = np.clip(np.expm1(m.predict(Xp)), 0, None)
                tgt = np.expm1(sub[c])
                ck = float((tgt * wp).sum() / max((ptr * wp).sum(), 1e-9))
                acc += ck * np.clip(np.expm1(m.predict(Xva)), 0, None)
                if s == SEEDS[0]: print(f'    {nm}/{c}: c = {ck:.4f}', flush=True)
            R[f'{nm}_{s}'] = np.log1p(acc)
        if s == SEEDS[0]:
            print(f'  комбинатор H {np.array2string(ah, precision=3)} · '
                  f'S {np.array2string(as_, precision=3)}', flush=True)
        print(f'  сид {s} за {time.perf_counter()-t0:.0f}с', flush=True)

    o = np.load(O / f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), **R})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0']}).join(t, on='user_id', how='inner')
    p0 = b['p0'].to_numpy(); zz = b['z'].to_numpy()
    sh = lambda k: float((zz - p0 * np.clip(b[k].to_numpy(), 0, None)).std())
    base = np.array([sh(f'A_{s}') for s in SEEDS])
    print(f'  A контроль      ' + ' '.join(f'{x:.5f}' for x in base), flush=True)
    for nm, lab in (('H', 'горизонт комб.'), ('HA', 'горизонт сумма'),
                    ('S', 'канал комб.'), ('SA', 'канал сумма')):
        f = np.array([sh(f'{nm}_{s}') for s in SEEDS]); d = base - f
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  {nm} {lab:<15}' + ' '.join(f'{x:.5f}' for x in f) +
              f' · Δ {d.mean():+.5f} · t {d.mean()/max(se,1e-12):+.2f}', flush=True)
    del X, Xva, Xp; gc.collect()
print('\nготово', flush=True)
