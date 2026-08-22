"""Percentile/rank: перепроверка шестью парными сидами на 318 и 348.

Якорь 378 НЕ ТРОГАЕТСЯ до принятия решения. Сегодня мы дважды
прочитали смысл в одиночном замере на 378, поэтому он оставлен
нетронутым подтверждающим якорем.

Сравнение строго парное: base и base+pct обучаются с одним сидом и
всеми одинаковыми параметрами, разность берётся внутри сида, поэтому
общий сидовый уровень сокращается.

Порог объявлен ДО запуска: pooled paired mean должен быть не ниже
+1.5e-4 при t не ниже примерно 1.5, иначе ветка закрывается.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from ecup import (SplitConfig, ModelConfig, HurdleGBDT, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.percentile import add_percentiles, PCT_COLS

ANCH = (318, 348); SEEDS = (42, 7, 2026, 13, 99, 123)
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
D = []
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    val = build_anchor(df, A, sp, None); z = np.log1p(val.y)
    va_id = np.full(len(val.y), A)
    print(f'\n=== якорь {A} · обучающие {an} · строк {len(y):,} ===', flush=True)
    P = {}
    for name, tr, va in (('base', Xd, val.X),
                         ('pct', add_percentiles(Xd, aid),
                          add_percentiles(val.X, va_id))):
        Xtr, feats = to_matrix(tr); Xva, _ = to_matrix(va, feats)
        P[name] = []
        t0 = time.perf_counter()
        for s in SEEDS:
            m = HurdleGBDT(config=ModelConfig(seed=s)).fit(
                Xtr, y, feature_names=feats, sample_weight=w, z_offset=zo, clf_init=ci)
            P[name].append(float((z - np.log1p(m.predict(
                Xva, p_target=last.p_bar, m_offset=last.l_plus))).std()))
            del m; gc.collect()
        print(f'  {name:<5} признаков {len(feats):>4} · ' +
              ' '.join(f'{v:.5f}' for v in P[name]) +
              f' · {time.perf_counter()-t0:.0f}с', flush=True)
        del Xtr, Xva; gc.collect()
    d = np.array(P['base']) - np.array(P['pct']); D.append(d)
    print(f'  разности ' + ' '.join(f'{v:+.5f}' for v in d) +
          f' · среднее {d.mean():+.5f}', flush=True)
    del Xd, val; gc.collect()

d = np.concatenate(D); se = d.std(ddof=1)/np.sqrt(len(d))
print(f'\n{"="*60}\npooled по {len(d)} парам: среднее {d.mean():+.5f} · '
      f'std {d.std(ddof=1):.5f} · SE {se:.5f} · t {d.mean()/se:+.2f}')
for A, x in zip(ANCH, D):
    print(f'  якорь {A}: {x.mean():+.5f}')
ok = d.mean() >= 1.5e-4 and d.mean()/se >= 1.5
print(f'\nобъявленный порог (+1.5e-4 и t >= 1.5): '
      f'{"ПРОЙДЕН — подтверждать на 378" if ok else "НЕ ПРОЙДЕН — ветка закрыта"}')
print(f'признаков в семье: {len(PCT_COLS)}')
