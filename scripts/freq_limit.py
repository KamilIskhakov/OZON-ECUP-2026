"""Два замера про частоту.

(1) НАСКОЛЬКО ВООБЩЕ ПРЕДСКАЗУЕМО N ПРИ ИЗВЕСТНОМ N > 0.

Счётчик на всех пользователях дал corr(s, logit p0) = 0.989, то есть
богатая разметка 0,1,2,... не породила другого упорядочивания, чем
бит. Остаётся выяснить, в чём ограничение: в доставке информации или
в том, что признаки просто не различают «купит один раз» и «купит
пять раз». Обучаем счётчик ТОЛЬКО НА ПОКУПАТЕЛЯХ и смотрим R^2 по
log(1+N) среди истинных покупателей валидационного якоря. Для
масштаба рядом печатается AUC текущего p0 на бите.

Если R^2 окажется низким, ветка частоты закрыта ПО СУЩЕСТВУ — это
вывод о данных, а не о конструкции.

(2) СТРАТИФИКАЦИЯ ВЫИГРЫША ФАКТОРИЗАЦИИ ПРЯМО ПО N_buy И AOV.

Ранее выигрыш был локализован по ИСТИННОМУ z, а связь с числом
заказов оставалась объяснением, а не замером. Здесь считаем вклад в
улучшение MSE по стратам N_buy = 1, 2, 3-4, 5+ и по квинтилям
AOV = Y / N_buy.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
import lightgbm as lgb

O = Path('artifacts/neural'); AN = (348, 378); SEED = 42
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)


def ndays(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + 30))
              .group_by('user_id')
              .agg(pl.col('d').filter(pl.col('gmv') > 0).n_unique().alias('N')))


print('=== (1) предсказуемость N при N > 0 ===', flush=True)
NV = {}
for A in AN:
    tr = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, tr, sp, None, verbose=False)
    w = anchor_weights(aid)
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    N = (pl.concat([key.filter(pl.col('_a') == a).join(ndays(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row')['N'].fill_null(0).to_numpy().astype('float64'))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = N > 0.5
    p = dict(ModelConfig(seed=SEED).reg_params)
    p.update(objective='poisson', n_estimators=600, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    t0 = time.perf_counter()
    m = lgb.LGBMRegressor(random_state=SEED, **p).fit(X[pos], N[pos], sample_weight=w[pos])
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    nv = (pl.DataFrame({'user_id': val.X['user_id'].to_numpy()})
            .join(ndays(A), on='user_id', how='left')['N'].fill_null(0).to_numpy().astype('float64'))
    lam = np.clip(m.predict(Xva), 1e-6, None)
    NV[A] = (val.X['user_id'].to_numpy(), nv, lam)
    b = nv > 0.5
    a_, f_ = np.log1p(nv[b]), np.log1p(lam[b])
    r2 = 1 - ((a_ - f_)**2).mean() / a_.var()
    o = np.load(O / f'oofpm_a{A}.npz')
    jj = pl.DataFrame({'user_id': o['user_id'], 'p0': o['p0'].astype('float64')}).join(
         pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), 'c': (nv > .5).astype('float64')}),
         on='user_id', how='inner')
    pp, cc = jj['p0'].to_numpy(), jj['c'].to_numpy()
    order = np.argsort(pp); rk = np.empty(len(pp)); rk[order] = np.arange(len(pp))
    n1 = cc.sum(); n0 = len(cc) - n1
    auc = (rk[cc > .5].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
    print(f'  якорь {A}: покупателей {b.mean():.3f} · среднее N|N>0 {nv[b].mean():.2f} · '
          f'std log(1+N) {a_.std():.3f}', flush=True)
    print(f'    R^2 по log(1+N) среди покупателей {r2:+.4f} · '
          f'corr {np.corrcoef(a_, f_)[0,1]:+.4f} · для масштаба AUC(p0) на бите {auc:.4f} · '
          f'{time.perf_counter()-t0:.0f}с', flush=True)
    del X, Xva; gc.collect()

print('\n=== (2) выигрыш факторизации по N_buy и AOV ===', flush=True)
F = np.load(O / 'freq_decomp.npz')
for A in AN:
    uid, nv, _ = NV[A]
    z = F[f'z_{A}']; p0 = F[f'p0_{A}']
    g = lambda k: np.log1p(np.exp(np.clip(F[f'{k}_{A}'], -20, 20)))
    ec = z - p0 * g('ctl400'); ef = z - p0 * g('buy200')
    dc = (ec - ec.mean())**2 - (ef - ef.mean())**2
    n = pl.DataFrame({'user_id': uid, 'N': nv}).join(
        pl.DataFrame({'user_id': F[f'uid_{A}'] if f'uid_{A}' in F.files else uid,
                      '_i': np.arange(len(z))}), on='user_id', how='inner')
    idx = n['_i'].to_numpy(); NN = n['N'].to_numpy()
    d2 = dc[idx]; z2 = z[idx]; tot = dc.mean()
    print(f'  якорь {A} · Δ shape {ec.std()-ef.std():+.5f}', flush=True)
    print(f'    {"страта":<20}{"доля":>8}{"вклад":>13}{"доля вклада":>13}')
    for lab, m in (('N=0', NN < .5), ('N=1', NN == 1), ('N=2', NN == 2),
                   ('N=3-4', (NN >= 3) & (NN <= 4)), ('N>=5', NN >= 5)):
        print(f'    {lab:<20}{m.mean()*len(idx)/len(z):>8.3f}'
              f'{(d2*m).sum()/len(z):>13.6f}{(d2*m).sum()/len(z)/tot:>13.1%}', flush=True)
    b = NN > .5
    aov = np.zeros(len(NN)); aov[b] = np.expm1(z2[b]) / NN[b]
    q = np.quantile(aov[b], np.linspace(0, 1, 6))
    print(f'    --- по квинтилям AOV = Y / N среди покупателей')
    for j in range(5):
        m = b & (aov >= q[j]) & (aov < q[j+1] if j < 4 else aov <= q[j+1])
        print(f'    AOV {q[j]:7.1f}..{q[j+1]:7.1f}{m.mean()*len(idx)/len(z):>8.3f}'
              f'{(d2*m).sum()/len(z):>13.6f}{(d2*m).sum()/len(z)/tot:>13.1%}', flush=True)
print('\nготово', flush=True)
