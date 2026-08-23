"""Систематический перебор параметров LightGBM, которые не проверялись.

Журнал подробно исследовал число деревьев, num_leaves, growth policy и
семейства моделей, но min_child_samples, max_bin, feature_fraction,
bagging_fraction и lambda_l1 были прямо перечислены как непроверенные.
На фоне TabM, PatchFormer и массивного предобучения это единственная
совсем нетронутая ось.

Ожидания низкие. Но замер дёшев и, в отличие от всего найденного за
сегодня, по построению НЕ коррелирован с годовым фильтром: он меняет
не информацию, а способ её нарезки.

Меняется ровно один параметр за раз от боевой конфигурации, оба якоря,
два сида, ранняя остановка боевая. Порог осмысленности задан заранее:
после калибровки переноса 0.39 кандидат должен дать не меньше 5e-4
офлайн, иначе он неразличим на лидерборде.
"""
import sys, warnings, gc, time, os; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT

SEEDS = (42, 7); HIST = 300
GRID = [('база', {})]
for v in (50, 100, 400, 800):
    GRID.append((f'min_child={v}', {'min_child_samples': v}))
for v in (63, 127, 511):
    GRID.append((f'max_bin={v}', {'max_bin': v}))
for v in (0.5, 0.65, 0.95):
    GRID.append((f'feat_frac={v}', {'feature_fraction': v}))
for v in (0.6, 1.0):
    GRID.append((f'bag_frac={v}', {'bagging_fraction': v}))
for v in (0.5, 2.0, 10.0):
    GRID.append((f'l1={v}', {'lambda_l1': v}))

df = load_panel()
OUT = {}
for A in (348, 378):
    sp = SplitConfig(max_history=HIST, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    z = np.log1p(val.y)
    print(f'\n=== ЯКОРЬ {A} · якорей {len(an)} · строк {len(y):,} ===', flush=True)
    for tag, over in GRID:
        vs = []
        for s in SEEDS:
            t0 = time.perf_counter()
            mc = ModelConfig(seed=s)
            mc.clf_params = {**mc.clf_params, **over}
            mc.reg_params = {**mc.reg_params, **over}
            hm = HurdleGBDT(config=mc).fit(X, y, feature_names=feats, sample_weight=w,
                                           z_offset=zo, clf_init=ci)
            p, m_ = hm.predict_parts(Xva, p_target=last.p_bar, m_offset=last.l_plus)
            vs.append(float((z - np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))).std()))
        OUT[(A, tag)] = np.array(vs)
        b = OUT[(A, 'база')]
        print(f'  {tag:<16}{" ".join(f"{x:.5f}" for x in vs)} · среднее {np.mean(vs):.5f}'
              f' · Δ {b.mean()-np.mean(vs):+.5f}'
              f' · парно {" ".join(f"{x:+.5f}" for x in b-np.array(vs))}'
              f' · {time.perf_counter()-t0:.0f}с', flush=True)
    del X, Xva; gc.collect()

print(f'\n{"="*78}\n{"вариант":<16}{"Δ 348":>11}{"Δ 378":>11}{"среднее":>11}{"знак":>8}')
rows = []
for tag, _ in GRID:
    if tag == 'база': continue
    d1 = OUT[(348, 'база')].mean() - OUT[(348, tag)].mean()
    d2 = OUT[(378, 'база')].mean() - OUT[(378, tag)].mean()
    rows.append((tag, d1, d2, (d1 + d2) / 2, 'оба +' if d1 > 0 and d2 > 0
                 else ('оба -' if d1 < 0 and d2 < 0 else 'РАЗНЫЙ')))
for tag, d1, d2, m, s in sorted(rows, key=lambda r: -r[3]):
    print(f'{tag:<16}{d1:>+11.5f}{d2:>+11.5f}{m:>+11.5f}{s:>8}')
print(f'\nпорог осмысленности 5e-4 офлайн (перенос 0.39)')
print('\nготово', flush=True)
