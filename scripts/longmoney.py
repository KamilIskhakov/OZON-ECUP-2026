"""v27_longmoney: рыночная нормировка + беcкэповая долгая история.

Проверка на 378 парным контролем при одинаковой конфигурации: меняется
РОВНО наличие нового блока признаков. Два сида, LGB-hurdle с боевой
ранней остановкой. Цель проверки — поймать катастрофу, а не измерить
эффект с точностью до пятого знака.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from ecup.market import market_features, _market

import os
A = int(os.environ.get('ANCHOR', 378)); SEEDS = (42, 7); HIST = 300
df = load_panel(); mkt = _market(df)
sp = SplitConfig(max_history=HIST, with_state=True)
an = [a for a in sp.train_anchors() if a + 30 <= A]
Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
X, feats = to_matrix(Xd)
uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()

t0 = time.perf_counter()
blocks, nm = [], None
for a in sorted(set(aid)):
    m = aid == a
    B, nm = market_features(df, int(a), uid_tr[m], mkt)
    blocks.append((m, B))
NEW = np.zeros((len(y), len(nm)), dtype='float32')
for m, B in blocks:
    NEW[m] = B
print(f'новых признаков {len(nm)} за {time.perf_counter()-t0:.0f}с: {nm}', flush=True)
for i, c in enumerate(nm):
    v = NEW[:, i]
    print(f'  {c:<18} среднее {v.mean():+8.3f} std {v.std():7.3f} '
          f'нулей {(np.abs(v) < 1e-12).mean():.3f}', flush=True)

val = build_anchor(df, A, sp, None)
Xva, _ = to_matrix(val.X, feats)
NEWva, _ = market_features(df, A, val.X['user_id'].to_numpy(), mkt)
z = np.log1p(val.y)

i_mkt = [i for i, c in enumerate(nm) if c.startswith('rel_')]
i_life = [i for i, c in enumerate(nm) if c.startswith('life_')]
VAR = {'база': (X, Xva, feats),
       '+рынок': (np.hstack([X, NEW[:, i_mkt]]), np.hstack([Xva, NEWva[:, i_mkt]]),
                  feats + [nm[i] for i in i_mkt]),
       '+жизнь': (np.hstack([X, NEW[:, i_life]]), np.hstack([Xva, NEWva[:, i_life]]),
                  feats + [nm[i] for i in i_life]),
       '+оба': (np.hstack([X, NEW]), np.hstack([Xva, NEWva]), feats + nm)}
X2, Xva2, f2 = VAR['+оба']
print(f'\nбазовых {X.shape[1]} · рынок {len(i_mkt)} · жизнь {len(i_life)}', flush=True)

res = {}
for tag, (Xt, Xv, ff) in VAR.items():
    for s in SEEDS:
        t0 = time.perf_counter()
        mc = ModelConfig(seed=s)
        hm = HurdleGBDT(config=mc).fit(Xt, y, feature_names=ff, sample_weight=w,
                                       z_offset=zo, clf_init=ci)
        p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
        zz = np.log1p(np.expm1(np.clip(p * np.clip(m_, 0, None), 0, 50)))
        from ecup import hurdle_glue
        zz = np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))
        res[(tag, s)] = float((z - zz).std())
        print(f'  {tag:<6} сид {s}: shape {res[(tag,s)]:.5f} · '
              f'деревьев {hm.best_iters} · {time.perf_counter()-t0:.0f}с', flush=True)

b = np.array([res[('база', s)] for s in SEEDS])
print(f'\n=== ЯКОРЬ {A} ===')
print(f'{"вариант":<10}{"сиды":>20}{"среднее":>11}{"Δ к базе":>11}{"парно":>20}')
print(f'{"база":<10}{" ".join(f"{x:.5f}" for x in b):>20}{b.mean():>11.5f}')
for tag in ('+рынок', '+жизнь', '+оба'):
    n = np.array([res[(tag, s)] for s in SEEDS])
    print(f'{tag:<10}{" ".join(f"{x:.5f}" for x in n):>20}{n.mean():>11.5f}'
          f'{b.mean()-n.mean():>+11.5f}{" ".join(f"{x:+.5f}" for x in b-n):>20}')

# важность новых признаков в обеих головах
mc = ModelConfig(seed=42)
hm = HurdleGBDT(config=mc).fit(X2, y, feature_names=f2, sample_weight=w,
                               z_offset=zo, clf_init=ci)
imp = dict((c, (a_, b_)) for c, a_, b_ in hm.importances(top=len(f2)))
tot_c = sum(v[0] for v in imp.values()); tot_r = sum(v[1] for v in imp.values())
new_c = sum(imp[c][0] for c in nm if c in imp); new_r = sum(imp[c][1] for c in nm if c in imp)
print(f'\nдоля важности нового блока: классификатор {new_c/max(tot_c,1e-9):.3f} · '
      f'регрессия {new_r/max(tot_r,1e-9):.3f} (при доле признаков {len(nm)/len(f2):.3f})')
rank = sorted(nm, key=lambda c: -(imp.get(c, (0, 0))[0] + imp.get(c, (0, 0))[1]))
for c in rank[:8]:
    a_, b_ = imp.get(c, (0, 0))
    print(f'  {c:<18} clf {a_:.4f} reg {b_:.4f}')
print('\nготово', flush=True)
