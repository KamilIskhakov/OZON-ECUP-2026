"""Временная абляция новых признаков: base против base + X.

Протокол: одна ФИКСИРОВАННАЯ модель, четыре якоря, семьи признаков
проверяются раздельно. Смена learner одновременно с признаками
запрещена, поэтому гиперпараметры LightGBM не трогаются вообще.

Признаки считаются ПО КАЖДОМУ ЯКОРЮ отдельно, включая обучающие:
признак пользователя на якоре a должен видеть только дни < a.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, HurdleGBDT, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.shocks import shock_betas
from ecup.intent import intent_stock

ANCH = (288, 318, 348, 378); O = Path('artifacts/neural')
FAM = {'shock': shock_betas, 'intent': intent_stock}
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)


def attach(Xd, aid, fn):
    """Присоединить признаки семьи, считая их для каждого якоря свои."""
    if fn is None:
        return Xd
    Z = Xd.with_columns(_aid=pl.Series(aid), _row=pl.int_range(pl.len(), dtype=pl.UInt32))
    parts = [Z.filter(pl.col('_aid') == a).join(fn(df, a), on='user_id', how='left')
             for a in sorted(set(aid))]
    return (pl.concat(parts, how='vertical_relaxed')
              .sort('_row').drop('_aid', '_row').fill_null(0.0))


res = {}
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    val = build_anchor(df, A, sp, None); z = np.log1p(val.y)
    print(f'\n=== якорь {A} · обучающие {an} · строк {len(y):,} ===', flush=True)
    for name, fn in (('base', None), *FAM.items()):
        Xtr, feats = to_matrix(attach(Xd, aid, fn))
        Xva, _ = to_matrix(attach(val.X, np.full(len(val.y), A), fn), feats)
        t0 = time.perf_counter()
        m = HurdleGBDT(config=ModelConfig(seed=42)).fit(
            Xtr, y, feature_names=feats, sample_weight=w, z_offset=zo, clf_init=ci)
        zp = np.log1p(m.predict(Xva, p_target=last.p_bar, m_offset=last.l_plus))
        res[(A, name)] = float((z - zp).std())
        print(f'  {name:<8} признаков {len(feats):>4} · shape {res[(A,name)]:.5f} · '
              f'{time.perf_counter()-t0:.0f}с', flush=True)
        del m, Xtr, Xva; gc.collect()
    del Xd, val; gc.collect()

print(f'\n{"="*54}\n{"якорь":>7}{"base":>10}' + ''.join(f'{k:>20}' for k in FAM))
for A in ANCH:
    row = f'{A:>7}{res[(A,"base")]:>10.5f}'
    for k in FAM:
        d = res[(A, 'base')] - res[(A, k)]
        row += f'{res[(A,k)]:>12.5f}{d:>+8.5f}'
    print(row)
print(f'\n{"среднее":>7}{"":>10}' + ''.join(
    f'{"":>12}{np.mean([res[(A,"base")]-res[(A,k)] for A in ANCH]):>+8.5f}' for k in FAM))
for k in FAM:
    s = [res[(A,'base')]-res[(A,k)] for A in ANCH]
    print(f'  {k}: знаки {" ".join("+" if x>0 else "-" for x in s)}')
