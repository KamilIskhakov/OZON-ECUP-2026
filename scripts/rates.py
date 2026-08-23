"""Тест механизма и чистый блок нормировок.

Пять вариантов при прочем равном, два сида, два якоря:

  база          183 признака как есть
  +r_capped     ОДИН признак log(1 + GMV_last300 / min(300, T+1))
  +r_full       ОДИН признак log(1 + GMV_0:T / (T+1))
  +чистый full  10 нормировок по всей истории
  +чистый оба   те же 10 плюс 10 по последним 300 дням

Разделение механизмов. Если +r_capped даёт большую часть эффекта,
главным был дрейф ЕДИНИЦ ИЗМЕРЕНИЯ: gmv_hist = 600 после 198 дней и
после 300 дней — разные состояния, а модель получала одно число.
Если прирост появляется только у +r_full, значит в старой истории за
границей max_history есть новый сигнал.

Ответ определяет v29: в первом случае надо нормировать ВСЕ
накопительные величины, во втором — расширять беcкэповый блок.
"""
import sys, warnings, gc, time, os; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from ecup.market import rate_features

A = int(os.environ.get('ANCHOR', 378)); SEEDS = (42, 7); HIST = 300
df = load_panel()
sp = SplitConfig(max_history=HIST, with_state=True)
an = [a for a in sp.train_anchors() if a + 30 <= A]
Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()

t0 = time.perf_counter(); NEW = None
for a in sorted(set(aid)):
    m = aid == a
    B, nm = rate_features(df, int(a), uid_tr[m])
    if NEW is None:
        NEW = np.zeros((len(y), len(nm)), dtype='float32')
    NEW[m] = B
val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
NEWva, _ = rate_features(df, A, val.X['user_id'].to_numpy())
z = np.log1p(val.y)
print(f'якорь {A} · признаков {len(nm)} за {time.perf_counter()-t0:.0f}с', flush=True)

i_f = [i for i, c in enumerate(nm) if c.startswith('f_')]
i_c = [i for i, c in enumerate(nm) if c.startswith('c_')]
i_rf = [nm.index('f_gmv_per_day')]; i_rc = [nm.index('c_gmv_per_day')]
print(f'  corr(f_gmv_per_day, c_gmv_per_day) '
      f'{np.corrcoef(NEW[:, i_rf[0]], NEW[:, i_rc[0]])[0,1]:+.4f}', flush=True)

VAR = {'база': ([], feats)}
for tag, idx in (('+r_capped', i_rc), ('+r_full', i_rf),
                 ('+чистый full', i_f), ('+чистый оба', list(range(len(nm))))):
    VAR[tag] = (idx, feats + [nm[i] for i in idx])

res = {}
for tag, (idx, ff) in VAR.items():
    Xt = X if not idx else np.hstack([X, NEW[:, idx]])
    Xv = Xva if not idx else np.hstack([Xva, NEWva[:, idx]])
    for s in SEEDS:
        t0 = time.perf_counter()
        hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
            Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
        from ecup import hurdle_glue
        p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
        res[(tag, s)] = float((z - np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))).std())
        print(f'  {tag:<14} сид {s}: {res[(tag,s)]:.5f} · {time.perf_counter()-t0:.0f}с',
              flush=True)
    if idx: del Xt, Xv; gc.collect()

b = np.array([res[('база', s)] for s in SEEDS])
print(f'\n=== ЯКОРЬ {A} ===')
print(f'{"вариант":<14}{"признаков":>10}{"сиды":>20}{"среднее":>11}{"Δ":>11}{"парно":>20}')
print(f'{"база":<14}{0:>10}{" ".join(f"{x:.5f}" for x in b):>20}{b.mean():>11.5f}')
for tag in ('+r_capped', '+r_full', '+чистый full', '+чистый оба'):
    n = np.array([res[(tag, s)] for s in SEEDS])
    print(f'{tag:<14}{len(VAR[tag][0]):>10}{" ".join(f"{x:.5f}" for x in n):>20}'
          f'{n.mean():>11.5f}{b.mean()-n.mean():>+11.5f}'
          f'{" ".join(f"{x:+.5f}" for x in b-n):>20}')
print('\nготово', flush=True)
