"""Глубина видимой истории как конфигурация, а не как признаки.

Блок life/rate работал потому, что max_history ОБРЕЗАЕТ историю, и
эффект одного признака GMV/(T+1) растёт с объёмом невидимых дней:
48 дней за границей на 348 дают +0.00046, 78 дней на 378 — +0.00122.
На боевом якоре 408 за границей лежит 108 дней.

Прямая проверка: не добавлять признаки, а снять обрезку у самих 183.
Боевой ансамбль использует 240/300/365; данные допускают 420, то есть
отсутствие обрезки вовсе.

Важно, что при allow_partial_history=True набор якорей от max_history
НЕ зависит (earliest_anchor = SELECTION_SPAN - 1), поэтому меняется
ровно глубина видимой истории, а не обучающая выборка.
"""
import sys, warnings, gc, time, os; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT

A = int(os.environ.get('ANCHOR', 378)); SEEDS = (42, 7); HS = (240, 300, 365, 420)
df = load_panel()
P = {}
for h in HS:
    sp = SplitConfig(max_history=h, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    z = np.log1p(val.y); uid = val.X['user_id'].to_numpy()
    i = {c: k for k, c in enumerate(feats)}
    hs = X[:, i['hist_span']]
    print(f'\n=== h={h} · якорей {len(an)} · признаков {len(feats)} · '
          f'hist_span медиана {np.median(hs):.0f} max {hs.max():.0f} ===', flush=True)
    for s in SEEDS:
        t0 = time.perf_counter()
        hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
            X, y, feature_names=feats, sample_weight=w, z_offset=zo, clf_init=ci)
        p, m_ = hm.predict_parts(Xva, p_target=last.p_bar, m_offset=last.l_plus)
        zz = np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))
        P[(h, s)] = (uid, zz)
        print(f'  сид {s}: shape {float((z-zz).std()):.5f} · деревьев {hm.best_iters} · '
              f'{time.perf_counter()-t0:.0f}с', flush=True)
    del X, Xva; gc.collect()

ref, zt = P[(HS[0], SEEDS[0])][0], None
o = np.load(f'artifacts/neural/oofpm_a{A}.npz')
key = pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y'])})
def al(u_, v_):
    return key.join(pl.DataFrame({'user_id': u_, 'p': v_}), on='user_id', how='left')['p'].to_numpy()
zt = key['z'].to_numpy()
Z = {k: al(*v) for k, v in P.items()}
sh = lambda v: float((zt - v).std())

print(f'\n=== ЯКОРЬ {A} · одиночные модели ===')
print(f'{"h":>6}{"сиды":>20}{"среднее":>11}{"Δ к h=300":>12}')
b = np.mean([sh(Z[(300, s)]) for s in SEEDS])
for h in HS:
    v = np.array([sh(Z[(h, s)]) for s in SEEDS])
    print(f'{h:>6}{" ".join(f"{x:.5f}" for x in v):>20}{v.mean():>11.5f}{b-v.mean():>+12.5f}')

print(f'\n=== ансамбли по три глубины (усреднение по сидам внутри) ===')
combos = [(240, 300, 365), (240, 300, 420), (240, 365, 420), (300, 365, 420),
          (240, 300, 365, 420)]
base = None
for c in combos:
    v = sh(np.mean([Z[(h, s)] for h in c for s in SEEDS], 0))
    if base is None: base = v
    print(f'  {str(c):<26}{v:.5f}{base-v:>+11.5f}')
print('\nготово', flush=True)
