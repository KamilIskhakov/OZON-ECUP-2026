"""Разрешение абляции: какой разброс дают сиды при неизменных признаках.

Все выводы вида «признак дал -0.0003» бессмысленны, если два запуска
одной и той же модели с разными сидами расходятся на столько же.
Сравнение ПАРНОЕ: для каждого сида обучаются base и base+X, разность
берётся внутри сида, поэтому общий сидовый уровень сокращается.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, HurdleGBDT, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.intent import intent_stock
from ecup.shocks import shock_betas

A = 378; SEEDS = (42, 7, 2026, 13, 99)
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
an = [a for a in sp.train_anchors() if a + 30 <= A]
Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
val = build_anchor(df, A, sp, None); z = np.log1p(val.y)


def attach(X, ids, fn):
    if fn is None: return X
    Z = X.with_columns(_aid=pl.Series(ids), _row=pl.int_range(pl.len(), dtype=pl.UInt32))
    p = [Z.filter(pl.col('_aid') == a).join(fn(df, a), on='user_id', how='left')
         for a in sorted(set(ids))]
    return pl.concat(p, how='vertical_relaxed').sort('_row').drop('_aid', '_row').fill_null(0.0)


M = {}
for name, fn in (('base', None), ('shock', shock_betas), ('intent', intent_stock)):
    Xtr, feats = to_matrix(attach(Xd, aid, fn))
    Xva, _ = to_matrix(attach(val.X, np.full(len(val.y), A), fn), feats)
    M[name] = []
    for s in SEEDS:
        m = HurdleGBDT(config=ModelConfig(seed=s)).fit(
            Xtr, y, feature_names=feats, sample_weight=w, z_offset=zo, clf_init=ci)
        M[name].append(float((z - np.log1p(m.predict(
            Xva, p_target=last.p_bar, m_offset=last.l_plus))).std()))
        del m; gc.collect()
    print(f'{name:<8} ' + ' '.join(f'{v:.5f}' for v in M[name]) +
          f'  · std по сидам {np.std(M[name]):.5f}', flush=True)
    del Xtr, Xva; gc.collect()

print(f'\n{"семья":<8}{"парная разность по сидам":>34}{"среднее":>10}{"std":>9}{"t":>7}')
for k in ('shock', 'intent'):
    d = np.array(M['base']) - np.array(M[k])
    t = d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1) > 0 else np.nan
    print(f'{k:<8}' + ' '.join(f'{v:+.5f}' for v in d) +
          f'{d.mean():>10.5f}{d.std(ddof=1):>9.5f}{t:>7.2f}')
