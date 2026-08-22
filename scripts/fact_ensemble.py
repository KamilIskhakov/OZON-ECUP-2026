"""Факторизованная голова во ВСЕХ LGB-членах бэкбона, поверх точного v23.

Голова выиграла у production на паре p·m: +0.00037/+0.00048 на 348 и
+0.00031/+0.00026 на 378, двадцать парных разностей из двадцати
положительны. Остаётся главный вопрос: сохранится ли это на ансамбле
и — решающее — ПОВЕРХ v23, где годовой фильтр и Gap-GRU могли уже
исправлять часть той же ошибки суммы.

Бэкбон: 6 LGB (история 240/300/365 × сиды 42/7) и 12 CatBoost, 0.4/0.6.
CatBoost-часть берётся сохранённой; шесть LGB переобучаются целиком,
включая СТАРУЮ голову, — иначе контроль сравнивался бы с чужим прогоном.
Классификатор общий для старой и факторизованной версии: различается
ровно голова регрессии при равном бюджете 600 против 300+300.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
import lightgbm as lgb

A = 378; H = 30; O = Path('artifacts/neural')
HIST = (240, 300, 365); SEEDS = (42, 7); BUDGET = 600
df = load_panel()


def counts(a):
    return (df.filter(pl.col('d').is_between(a + 1, a + H)).group_by('user_id')
              .agg(n_ord=pl.col('to_ord').sum().cast(pl.Float64),
                   n_buy=(pl.col('gmv') > 0).sum().cast(pl.Float64)))


def reg(X, y, w, n, seed):
    p = dict(ModelConfig(seed=seed).reg_params)
    p.update(n_estimators=n, verbose=-1, n_jobs=-1)
    p.pop('early_stopping_rounds', None)
    return lgb.LGBMRegressor(random_state=seed, **p).fit(X, y, sample_weight=w)


wmean = lambda v, w: float((v * w).sum() / w.sum())
acc = {k: [] for k in ('old', 'buy', 'ord')}
for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
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
    uid = val.X['user_id'].to_numpy()
    Xp, wp, ap = X[pos], w[pos], aid[pos]
    zp = np.log1p(y[pos]) - zo[pos]; lY = np.log(y[pos]); la = max(an)
    print(f'\n=== история {h} · якорей {len(an)} · строк {len(y):,} ===', flush=True)

    for s in SEEDS:
        t0 = time.perf_counter()
        cp = dict(ModelConfig(seed=s).clf_params)
        cp.update(n_estimators=BUDGET, verbose=-1, n_jobs=-1)
        cp.pop('early_stopping_rounds', None)
        clf = lgb.LGBMClassifier(random_state=s, **cp).fit(
            X, pos.astype(np.int8), sample_weight=w, init_score=ci)
        raw = clf.predict(Xva, raw_score=True)
        p = 1 / (1 + np.exp(-(raw + np.log(last.p_bar / (1 - last.p_bar)))))
        m_old = reg(Xp, zp, wp, BUDGET, s).predict(Xva) + last.l_plus
        acc['old'].append((uid, np.log1p(hurdle_glue(p, np.clip(m_old, 0, None)))))
        for nm, num in (('buy', C['n_buy'].to_numpy()[pos]),
                        ('ord', C['n_ord'].to_numpy()[pos])):
            u = np.log(np.clip(num, 1.0, None)); v = lY - u
            mu_u = {a: wmean(u[ap == a], wp[ap == a]) for a in sorted(set(ap))}
            mu_v = {a: wmean(v[ap == a], wp[ap == a]) for a in sorted(set(ap))}
            ou = np.array([mu_u[a] for a in ap]); ov = np.array([mu_v[a] for a in ap])
            ss = (reg(Xp, u - ou, wp, BUDGET // 2, s).predict(Xva) + mu_u[la] +
                  reg(Xp, v - ov, wp, BUDGET // 2, s).predict(Xva) + mu_v[la])
            mf = np.log1p(np.exp(np.clip(ss, -20, 20)))
            acc[nm].append((uid, np.log1p(hurdle_glue(p, np.clip(mf, 0, None)))))
        print(f'  сид {s} за {time.perf_counter()-t0:.0f}с', flush=True)
    del X, Xva, Xp; gc.collect()

# --- сведение к общему порядку пользователей и оценка поверх точного v23
o = np.load(O / f'oofpm_a{A}.npz'); ref = o['user_id']
def align(lst):
    out = []
    for uid, z in lst:
        t = pl.DataFrame({'user_id': uid, 'z': z})
        out.append(pl.DataFrame({'user_id': ref}).join(t, on='user_id', how='left')['z']
                   .to_numpy())
    return np.mean(out, 0)

S = '/tmp/'
E = np.load(S + 'cb_ens2.npz')
zc = np.mean([E[k] for k in E.files if k.startswith(('cb_', 'brd', 'dpw'))], 0)
d16 = np.load(O / f'dz_a{A}.npz')['dz']; dann = np.load(S + f'd_annual_{A}.npy')
CORR = 0.35 * (d16 - d16.mean()) + 0.0104 * (dann - dann.mean())
z = np.log1p(o['y'])
zl = {k: align(v) for k, v in acc.items()}
zl_saved = np.mean([E[k] for k in E.files if k.startswith('lgb')], 0)

print(f'\n{"="*62}\n{"вариант LGB":<22}{"только LGB":>12}{"бэкбон":>10}{"+ v23":>10}{"Δ":>10}')
b0 = None
for k in ('old', 'buy', 'ord'):
    s_l = (z - zl[k]).std()
    s_b = (z - (0.4 * zl[k] + 0.6 * zc)).std()
    s_v = (z - (0.4 * zl[k] + 0.6 * zc + CORR)).std()
    if b0 is None: b0 = s_v
    print(f'{k:<22}{s_l:>12.5f}{s_b:>10.5f}{s_v:>10.5f}{b0-s_v:>+10.5f}')
s_saved = (z - (0.4 * zl_saved + 0.6 * zc + CORR)).std()
print(f'{"сохранённый v23":<22}{(z-zl_saved).std():>12.5f}'
      f'{(z-(0.4*zl_saved+0.6*zc)).std():>10.5f}{s_saved:>10.5f}')
np.savez_compressed(O / 'fact_ensemble.npz', **{f'zl_{k}': v for k, v in zl.items()},
                    user_id=ref)
print('\nготово', flush=True)
