"""Пакет вариантов головы регрессии: замена, гибрид, learned features, Poisson.

Полная замена головы дала +0.0004 на модели, но после усреднения шести
LGB и веса 0.4 сжалась до +0.00006. Диагноз не «факторизация не
работает», а «эффект разбавляется ансамблем». Поэтому проверяются
конструкции, которые НЕ выбрасывают прямую голову:

  A  m_direct               600            контроль
  B  m_fact                 300+300        полная замена
  C  m_direct + lam·(m_fact - m_direct)    гибрид, lam заморожен
  D  LGB(X, n_hat, a_hat)   600            факторизация как ДВА ПРИЗНАКА
  E  как B, но счётчик Poisson вместо L2 на log N

D — главный: прямая голова сохраняет верную цель log(1+Y), но
получает две координаты, которые уже разложили сложность на частоту
и величину. Признаки строятся leave-one-anchor-out: для строк якоря A
пара обучена БЕЗ него, иначе утечка таргета.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
import lightgbm as lgb

ANCH = (348, 378); H = 30; O = Path('artifacts/neural')
SEEDS = (42, 7, 2026); BUDGET = 600
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)


def counts(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + H)).group_by('user_id')
              .agg(n_buy=(pl.col('gmv') > 0).sum().cast(pl.Float64)))


def reg(X, y, w, n, s, obj=None):
    p = dict(ModelConfig(seed=s).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    if obj: p['objective'] = obj
    return lgb.LGBMRegressor(random_state=s, **p).fit(X, y, sample_weight=w)


wmean = lambda v, w: float((v * w).sum() / w.sum())
LAM = {}
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    C = (pl.concat([key.filter(pl.col('_a') == a).join(counts(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row').fill_null(0.0))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = y > 0
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Xp, wp, ap = X[pos], w[pos], aid[pos]
    zp = np.log1p(y[pos]) - zo[pos]; lY = np.log(y[pos]); la = max(an)
    nb = C['n_buy'].to_numpy()[pos]
    u = np.log(np.clip(nb, 1.0, None)); v = lY - u
    mu_u = {a: wmean(u[ap == a], wp[ap == a]) for a in sorted(set(ap))}
    mu_v = {a: wmean(v[ap == a], wp[ap == a]) for a in sorted(set(ap))}
    ou = np.array([mu_u[a] for a in ap]); ov = np.array([mu_v[a] for a in ap])
    print(f'\n=== якорь {A} · обучающие {an} · покупателей {pos.mean():.3f} ===', flush=True)

    R = {}
    for s in SEEDS:
        t0 = time.perf_counter()
        R[f'A_{s}'] = reg(Xp, zp, wp, BUDGET, s).predict(Xva) + last.l_plus
        fu = reg(Xp, u - ou, wp, BUDGET // 2, s); fv = reg(Xp, v - ov, wp, BUDGET // 2, s)
        sB = fu.predict(Xva) + mu_u[la] + fv.predict(Xva) + mu_v[la]
        R[f'B_{s}'] = np.log1p(np.exp(np.clip(sB, -20, 20)))
        # E: счётчик как Poisson на самом N, а не L2 на log N
        fp = reg(Xp, nb, wp, BUDGET // 2, s, obj='poisson')
        sE = np.log(np.clip(fp.predict(Xva), 1e-3, None)) + fv.predict(Xva) + mu_v[la]
        R[f'E_{s}'] = np.log1p(np.exp(np.clip(sE, -20, 20)))
        # D: OOF-признаки leave-one-anchor-out, чтобы не было утечки таргета
        nh = np.empty(pos.sum()); ah = np.empty(pos.sum())
        for a in sorted(set(ap)):
            m = ap != a; k = ap == a
            nh[k] = reg(Xp[m], (u - ou)[m], wp[m], BUDGET // 4, s).predict(Xp[k])
            ah[k] = reg(Xp[m], (v - ov)[m], wp[m], BUDGET // 4, s).predict(Xp[k])
        nv = fu.predict(Xva); av = fv.predict(Xva)   # для вала — обучены на всех
        R[f'D_{s}'] = reg(np.column_stack([Xp, nh, ah]), zp, wp, BUDGET, s).predict(
            np.column_stack([Xva, nv, av])) + last.l_plus
        print(f'  сид {s} за {time.perf_counter()-t0:.0f}с', flush=True)

    o = np.load(O / f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), **R})
    b = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y']),
                      'p0': o['p0']}).join(t, on='user_id', how='inner')
    p0 = b['p0'].to_numpy(); zz = b['z'].to_numpy()
    sh = lambda arr: float((zz - p0 * np.clip(arr, 0, None)).std())
    base = np.array([sh(b[f'A_{s}'].to_numpy()) for s in SEEDS])
    print(f'  A (контроль) ' + ' '.join(f'{x:.5f}' for x in base), flush=True)
    for k, nm in (('B', 'полная замена'), ('D', 'два признака'), ('E', 'Poisson-счётчик')):
        f = np.array([sh(b[f'{k}_{s}'].to_numpy()) for s in SEEDS]); d = base - f
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'  {k} {nm:<16}' + ' '.join(f'{x:.5f}' for x in f) +
              f' · Δ {d.mean():+.5f} · t {d.mean()/max(se,1e-12):+.2f}', flush=True)
    # C: гибрид, коэффициент с ПРЕДЫДУЩЕГО якоря
    md = np.mean([b[f'A_{s}'].to_numpy() for s in SEEDS], 0)
    mf = np.mean([b[f'B_{s}'].to_numpy() for s in SEEDS], 0)
    e = zz - p0 * np.clip(md, 0, None); dd = p0 * (np.clip(mf, 0, None) - np.clip(md, 0, None))
    lam_own = float(((e - e.mean()) * (dd - dd.mean())).mean() / ((dd - dd.mean())**2).mean())
    print(f'  C гибрид: собственный lam {lam_own:+.4f}', flush=True)
    if LAM:
        lp = list(LAM.values())[-1]
        print(f'    с замороженным lam {lp:+.4f}: '
              f'{float((zz - (p0*np.clip(md,0,None) + lp*dd)).std()):.5f} '
              f'(контроль {e.std():.5f})', flush=True)
    LAM[A] = lam_own
    del X, Xva, Xp; gc.collect()
print('\nготово', flush=True)
