"""Боевое направление факторизованной головы на якоре 408.

Считается РАЗНОСТЬ двух бэкбонов, а не бэкбон заново: v23 остаётся
нетронутым, к нему прибавляется d = z_fact - z_old. Это проверено на
378 против СОХРАНЁННОГО v23 и дало +0.00006 при вкладе одной лишь
LGB-части, что совпало с замером на переобученном контроле.

CatBoost НЕ факторизуется: на якоре 348 buy дал -0.00013 (t = -2.44),
ord +0.00003 (t = +1.46) — против +0.00037 и +0.00048 у LightGBM.
Симметричные деревья CatBoost уже гладкие по построению, поэтому
разложение на две гладкие компоненты ничего не добавляет, а деление
бюджета пополам отнимает. Направление строится только по LGB, вес 0.4.

Направление демеанируется: уровень v23 подобран ручкой a_m по
лидерборду, и менять его непроверенным сдвигом нельзя. Валидировалась
форма — она и переносится.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.catboost_model import CatBoostConfig
import lightgbm as lgb

H = 30; O = Path('artifacts/neural'); HIST = (240, 300, 365); SEEDS = (42, 7)
BUDGET = 600; A_M = 0.18
df = load_panel()
FIN = SplitConfig(max_history=300, with_state=True).final_anchor
print(f'боевой якорь {FIN}', flush=True)


def counts(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + H)).group_by('user_id')
              .agg(n_ord=pl.col('to_ord').sum().cast(pl.Float64)))


def lgb_reg(X, y, w, n, s):
    p = dict(ModelConfig(seed=s).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1); p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(random_state=s, **p).fit(X, y, sample_weight=w)


def cb_reg(X, y, w, n, s):
    from catboost import CatBoostRegressor, Pool
    p = dict(CatBoostConfig(seed=s).reg_params); p['iterations'] = n
    m = CatBoostRegressor(random_seed=s, **p); m.fit(Pool(X, label=y, weight=w), verbose=False)
    return m


wmean = lambda v, w: float((v * w).sum() / w.sum())
D = {'lgb': []}
for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
    an = sp.refit_anchors()
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    key = pl.DataFrame({'user_id': Xd['user_id'].to_numpy(), '_a': aid,
                        '_row': np.arange(len(aid), dtype='uint32')})
    C = (pl.concat([key.filter(pl.col('_a') == a).join(counts(a), on='user_id', how='left')
                    for a in sorted(set(aid))], how='vertical_relaxed')
           .sort('_row').fill_null(0.0))
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    pos = y > 0
    fin = build_anchor(df, FIN, sp, None, with_target=False)
    Xte, _ = to_matrix(fin.X, feats); uid = fin.X['user_id'].to_numpy()
    Xp, wp, ap = X[pos], w[pos], aid[pos]
    zp = np.log1p(y[pos]) - zo[pos]; lY = np.log(y[pos]); la = max(an)
    num = C['n_ord'].to_numpy()[pos]
    u = np.log(np.clip(num, 1.0, None)); v = lY - u
    mu_u = {a: wmean(u[ap == a], wp[ap == a]) for a in sorted(set(ap))}
    mu_v = {a: wmean(v[ap == a], wp[ap == a]) for a in sorted(set(ap))}
    ou = np.array([mu_u[a] for a in ap]); ov = np.array([mu_v[a] for a in ap])
    p_base = last.p_bar
    print(f'\n=== история {h} · якорей {len(an)} · строк {len(y):,} ===', flush=True)

    for s in SEEDS:
        for fam in ('lgb',):
            t0 = time.perf_counter()
            if fam == 'lgb':
                cp = dict(ModelConfig(seed=s).clf_params)
                cp.update(n_estimators=BUDGET, verbose=-1, n_jobs=-1)
                cp.pop('early_stopping_rounds', None)
                clf = lgb.LGBMClassifier(random_state=s, **cp).fit(
                    X, pos.astype(np.int8), sample_weight=w, init_score=ci)
                raw = clf.predict(Xte, raw_score=True); R = lgb_reg
            else:
                from catboost import CatBoostClassifier, Pool
                cp = dict(CatBoostConfig(seed=s).clf_params); cp['iterations'] = BUDGET
                clf = CatBoostClassifier(random_seed=s, **cp)
                clf.fit(Pool(X, label=pos.astype(np.int8), weight=w, baseline=ci),
                        verbose=False)
                raw = clf.predict(Xte, prediction_type='RawFormulaVal'); R = cb_reg
            p = 1 / (1 + np.exp(-(raw + np.log(p_base / (1 - p_base)))))
            m_old = R(Xp, zp, wp, BUDGET, s).predict(Xte) + last.l_plus + A_M
            ss = (R(Xp, u - ou, wp, BUDGET // 2, s).predict(Xte) + mu_u[la] +
                  R(Xp, v - ov, wp, BUDGET // 2, s).predict(Xte) + mu_v[la] + A_M)
            m_f = np.log1p(np.exp(np.clip(ss, -20, 20)))
            zo_ = np.log1p(hurdle_glue(p, np.clip(m_old, 0, None)))
            zf_ = np.log1p(hurdle_glue(p, np.clip(m_f, 0, None)))
            D[fam].append((uid, zf_ - zo_))
            print(f'  {fam} сид {s} за {time.perf_counter()-t0:.0f}с · '
                  f'std направления {(zf_-zo_).std():.5f}', flush=True)
    del X, Xte, Xp; gc.collect()

ref = np.load(O / 'dz_prod_a408.npz')['user_id']
def align(lst):
    out = []
    for uid, d in lst:
        t = pl.DataFrame({'user_id': uid, 'd': d})
        out.append(pl.DataFrame({'user_id': ref}).join(t, on='user_id', how='left')['d']
                   .to_numpy())
    return np.mean(out, 0)

dl = align(D['lgb'])
d = 0.4 * dl; d = d - d.mean()
print(f'\nLGB std {dl.std():.5f}')
print(f'итоговое направление: std {d.std():.5f}, среднее {d.mean():+.2e}')
np.savez_compressed(O / 'fact_prod_a408.npz', user_id=ref, d=d, d_lgb=dl)
print('готово', flush=True)
