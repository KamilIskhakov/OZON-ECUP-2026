"""Боевое направление ПОЛНОЙ замены LGB-части на якоре 408.

v25 забрал только факторизационную половину: в fact_prod.py обе стороны
разности уже были с фиксированными 600 деревьями, поэтому переход
noES остался неиспользованным. Замер на 378 поверх точного v23:

    сохранённый v23 (ES)      1.67753
    old noES                  1.67746   +0.00007
    ord noES + факторизация   1.67739   +0.00014

Здесь считаются ТРИ прогноза на боевом якоре по той же сетке, что и в
production (история 240/300/365 x сиды 42/7, ровно шесть LGB):

    z_prod   production: ранняя остановка 100, eval_frac 0.12
    z_old    та же голова, но фиксированные 600 без ранней остановки
    z_alt    600 без ES + факторизованная голова ord (300+300)

Направление v26 = 0.4 (z_alt - z_prod), демеанированное: уровень v23
задан ручкой a_m = 0.18 по лидерборду, валидировалась форма.

Разложение на d_noES и d_fact печатается для сверки, а d_fact ещё и
сравнивается с сохранённым fact_prod_a408.npz — если совпадёт, значит
воспроизведение production-стороны корректно.

CatBoost не трогаем: там знак факторизации переворачивался между
якорями (buy -0.00013 на 348 против +0.00016 на 378).
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
import lightgbm as lgb

H = 30; O = Path('artifacts/neural'); HIST = (240, 300, 365); SEEDS = (42, 7)
BUDGET = 600; A_M = 0.18
df = load_panel()
FIN = SplitConfig(max_history=300, with_state=True).final_anchor
print(f'боевой якорь {FIN}', flush=True)


def counts(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + H)).group_by('user_id')
              .agg(n_ord=pl.col('to_ord').sum().cast(pl.Float64)))


def reg_noes(X, y, w, n, s):
    p = dict(ModelConfig(seed=s).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1); p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(random_state=s, **p).fit(X, y, sample_weight=w)


wmean = lambda v, w: float((v * w).sum() / w.sum())
acc = {k: [] for k in ('prod', 'old', 'alt')}
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
    p_base = last.p_bar; m_off = last.l_plus + A_M
    print(f'\n=== история {h} · якорей {len(an)} · строк {len(y):,} ===', flush=True)

    for s in SEEDS:
        t0 = time.perf_counter()
        # --- production: ранняя остановка, ровно как в make_submission
        mc = ModelConfig(seed=s)
        hm = HurdleGBDT(config=mc).fit(
            X, y, feature_names=feats, sample_weight=w, z_offset=zo, clf_init=ci,
            early_stopping_rounds=mc.early_stopping_rounds, eval_frac=mc.eval_frac,
            refit_full=mc.refit_full)
        z_prod = np.log1p(hm.predict(Xte, p_target=p_base, m_offset=m_off))
        bi = hm.best_iters
        # --- альтернативная часть: общий классификатор без ранней остановки
        cp = dict(mc.clf_params); cp.update(n_estimators=BUDGET, verbose=-1, n_jobs=-1)
        cp.pop('early_stopping_rounds', None)
        clf = lgb.LGBMClassifier(random_state=s, **cp).fit(
            X, pos.astype(np.int8), sample_weight=w, init_score=ci)
        raw = clf.predict(Xte, raw_score=True)
        p = 1 / (1 + np.exp(-(raw + np.log(p_base / (1 - p_base)))))
        m_old = reg_noes(Xp, zp, wp, BUDGET, s).predict(Xte) + m_off
        ss = (reg_noes(Xp, u - ou, wp, BUDGET // 2, s).predict(Xte) + mu_u[la] +
              reg_noes(Xp, v - ov, wp, BUDGET // 2, s).predict(Xte) + mu_v[la] + A_M)
        m_f = np.log1p(np.exp(np.clip(ss, -20, 20)))
        z_old = np.log1p(hurdle_glue(p, np.clip(m_old, 0, None)))
        z_alt = np.log1p(hurdle_glue(p, np.clip(m_f, 0, None)))
        for k, z_ in (('prod', z_prod), ('old', z_old), ('alt', z_alt)):
            acc[k].append((uid, z_))
        print(f'  сид {s} за {time.perf_counter()-t0:.0f}с · ES деревьев clf {bi[0]} reg {bi[1]} · '
              f'std z: prod {z_prod.std():.4f} old {z_old.std():.4f} alt {z_alt.std():.4f}',
              flush=True)
    del X, Xte, Xp; gc.collect()

ref = np.load(O / 'dz_prod_a408.npz')['user_id']
def align(lst):
    out = []
    for uid_, z_ in lst:
        t = pl.DataFrame({'user_id': uid_, 'z': z_})
        out.append(pl.DataFrame({'user_id': ref}).join(t, on='user_id', how='left')['z']
                   .to_numpy())
    return np.mean(out, 0)

Z = {k: align(v) for k, v in acc.items()}
d_noes = 0.4 * (Z['old'] - Z['prod'])
d_fact = 0.4 * (Z['alt'] - Z['old'])
d_tot = 0.4 * (Z['alt'] - Z['prod'])
d = d_tot - d_tot.mean()
print(f'\nразложение направления (в единицах вклада в z, вес 0.4 уже внутри):')
print(f'  noES  : std {d_noes.std():.5f} · среднее {d_noes.mean():+.5f}')
print(f'  fact  : std {d_fact.std():.5f} · среднее {d_fact.mean():+.5f}')
print(f'  всего : std {d_tot.std():.5f} · среднее {d_tot.mean():+.5f}')
old = np.load(O / 'fact_prod_a408.npz')
d_saved = 0.4 * old['d_lgb']
print(f'\nсверка факторизационной части с сохранённой (v25):')
print(f'  сохранённая std {d_saved.std():.5f} · новая std {d_fact.std():.5f} · '
      f'corr {np.corrcoef(d_saved, d_fact)[0,1]:+.4f}')
print(f'\nитоговое направление v26: std {d.std():.5f}, среднее {d.mean():+.2e}')
np.savez_compressed(O / 'altlgb_prod_a408.npz', user_id=ref, d=d, d_tot=d_tot,
                    d_noes=d_noes, d_fact=d_fact,
                    z_prod=Z['prod'], z_old=Z['old'], z_alt=Z['alt'])
print('готово', flush=True)
