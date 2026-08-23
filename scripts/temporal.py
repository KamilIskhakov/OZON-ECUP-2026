"""v29_temporal: временные веса раздельно по головам + временная ранняя остановка.

Две смены схемы обучения во времени, обе никогда не проверялись.

ПЕРВАЯ. Сейчас обе головы получают один и тот же вес
w(a) = 2^(-age/90), причём значение 90 никогда не выбиралось
систематически. Но сезонное изменение в отобранной популяции почти
целиком живёт в интенсивной марже m, тогда как p заметно стабильнее,
поэтому одинаковая скорость забывания у двух голов не обоснована.

    A  h_p=90   h_m=90    точный текущий контроль
    B  h_p=180  h_m=60
    C  h_p=inf  h_m=60
    D  h_p=180  h_m=45

ВТОРАЯ, важнее. Ранняя остановка сейчас идёт по СЛУЧАЙНЫМ 12 % строк
из объединённого трейна. Это ловит обычное переобучение, но не
соответствует задаче «прошлые месяцы -> следующий месяц»: сложность
выбирается на случайной подвыборке прошлого, а оценивается на будущем.

Временная схема для якоря A:
    обучение на якорях до предпоследнего, ранняя остановка на
    ПОСЛЕДНЕМ доступном якоре -> k_p, k_m;
    затем переобучение на всех якорях с фиксированными k -> прогноз A.

Обязательный контроль: A (текущие веса + временная ES) против BASE
(текущие веса + случайная ES). Он показывает, нужна ли вообще смена
half-life, или весь выигрыш идёт от правильного выбора числа деревьев.

Политика принимается, только если знак улучшения совпал на 348 и 378.
"""
import sys, warnings, gc, time, os; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
import lightgbm as lgb

O = Path('artifacts/neural'); SEEDS = (42, 7); HIST = 300; BIG = 3000
POL = {'A': (90., 90.), 'B': (180., 60.), 'C': (np.inf, 60.), 'D': (180., 45.)}


def wts(aid, h):
    if not np.isfinite(h):
        return np.ones(len(aid))
    return np.power(0.5, (aid.max() - aid).astype('float64') / h)


def fit_temporal(X, y, aid, ci, zo, es_a, s, hp, hm):
    """Ранняя остановка на последнем якоре, затем refit на всех с найденным k."""
    pos = y > 0
    tr = aid != es_a; es = aid == es_a
    mc = ModelConfig(seed=s)
    cp = {**mc.clf_params, 'n_estimators': BIG, 'verbose': -1, 'n_jobs': -1}
    rp = {**mc.reg_params, 'n_estimators': BIG, 'verbose': -1, 'n_jobs': -1}
    z = np.log1p(y) - zo
    wp_i, wm_i = wts(aid[tr], hp), wts(aid[tr], hm)
    clf = lgb.LGBMClassifier(random_state=s, **cp).fit(
        X[tr], pos[tr].astype(np.int8), sample_weight=wp_i, init_score=ci[tr],
        eval_set=[(X[es], pos[es].astype(np.int8))], eval_init_score=[ci[es]],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(100, verbose=False)])
    k_p = clf.best_iteration_ or BIG
    pt = tr & pos; pe = es & pos
    reg = lgb.LGBMRegressor(random_state=s, **rp).fit(
        X[pt], z[pt], sample_weight=wts(aid[pt], hm),
        eval_set=[(X[pe], z[pe])], eval_metric='l2',
        callbacks=[lgb.early_stopping(100, verbose=False)])
    k_m = reg.best_iteration_ or BIG
    # --- refit на ВСЕХ якорях с фиксированным числом деревьев
    clf = lgb.LGBMClassifier(random_state=s, **{**cp, 'n_estimators': k_p}).fit(
        X, pos.astype(np.int8), sample_weight=wts(aid, hp), init_score=ci)
    reg = lgb.LGBMRegressor(random_state=s, **{**rp, 'n_estimators': k_m}).fit(
        X[pos], z[pos], sample_weight=wts(aid[pos], hm))
    return clf, reg, k_p, k_m


R = {}
df = load_panel()
for A in (348, 378):
    sp = SplitConfig(max_history=HIST, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    z = np.log1p(val.y); uid = val.X['user_id'].to_numpy()
    es_a = max(an)
    print(f'\n=== ЯКОРЬ {A} · якоря {an} · ранняя остановка на {es_a} ===', flush=True)
    for s in SEEDS:
        t0 = time.perf_counter()
        hm_ = HurdleGBDT(config=ModelConfig(seed=s)).fit(
            X, y, feature_names=feats, sample_weight=anchor_weights(aid),
            z_offset=zo, clf_init=ci)
        p, m_ = hm_.predict_parts(Xva, p_target=last.p_bar, m_offset=last.l_plus)
        R[(A, 'BASE', s)] = np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))
        print(f'  BASE сид {s}: shape {float((z-R[(A,"BASE",s)]).std()):.5f} · '
              f'деревьев {hm_.best_iters} · {time.perf_counter()-t0:.0f}с', flush=True)
        for tag, (hp, hmm) in POL.items():
            t0 = time.perf_counter()
            clf, reg, k_p, k_m = fit_temporal(X, y, aid, ci, zo, es_a, s, hp, hmm)
            raw = clf.predict(Xva, raw_score=True)
            pp = 1 / (1 + np.exp(-(raw + np.log(last.p_bar / (1 - last.p_bar)))))
            mm = reg.predict(Xva) + last.l_plus
            R[(A, tag, s)] = np.log1p(hurdle_glue(pp, np.clip(mm, 0, None)))
            print(f'  {tag} (h_p={hp}, h_m={hmm}) сид {s}: '
                  f'shape {float((z-R[(A,tag,s)]).std()):.5f} · деревьев ({k_p}, {k_m}) · '
                  f'{time.perf_counter()-t0:.0f}с', flush=True)
    R[(A, 'z')] = z; R[(A, 'uid')] = uid
    del X, Xva; gc.collect()

print(f'\n{"="*72}\n{"вариант":<10}{"Δ 348":>11}{"Δ 378":>11}{"среднее":>11}{"знак":>9}')
res = []
for tag in ('A', 'B', 'C', 'D'):
    d = []
    for A in (348, 378):
        zz = R[(A, 'z')]
        b = np.mean([float((zz - R[(A, 'BASE', s)]).std()) for s in SEEDS])
        v = np.mean([float((zz - R[(A, tag, s)]).std()) for s in SEEDS])
        d.append(b - v)
    sg = 'оба +' if min(d) > 0 else ('оба -' if max(d) < 0 else 'РАЗНЫЙ')
    res.append((tag, d[0], d[1], np.mean(d), sg))
    print(f'{tag:<10}{d[0]:>+11.5f}{d[1]:>+11.5f}{np.mean(d):>+11.5f}{sg:>9}')
np.savez_compressed(O / 'temporal_val.npz',
                    **{f'{A}_{t}_{s}': R[(A, t, s)] for A in (348, 378)
                       for t in ('BASE', 'A', 'B', 'C', 'D') for s in SEEDS},
                    **{f'{A}_z': R[(A, 'z')] for A in (348, 378)},
                    **{f'{A}_uid': R[(A, 'uid')] for A in (348, 378)})
ok = [r for r in res if r[4] == 'оба +']
print(f'\nпрошли по знаку: {[r[0] for r in ok] or "НЕТ"}')
if ok:
    best = max(ok, key=lambda r: r[3])
    print(f'лучшая политика: {best[0]} · среднее {best[3]:+.5f}')
print('\nготово', flush=True)
