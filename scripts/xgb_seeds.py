"""Три сида direct-XGB: снижает ли усреднение шумовую часть направления.

Выигрыш направления есть C²/D. Усреднение по сидам убирает из d
независимый шум, не трогая систематическую часть, поэтому D должно
упасть сильнее, чем C. Один сид дал перенос +0.00013 при lam,
замороженном по 288/318/348; здесь проверяется, растёт ли он.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, load_panel, build_anchor, build_training_set,
                  to_matrix, anchor_weights)
from ecup.xgb_model import XGBConfig, DirectXGB

DEV = 'cuda' if len(sys.argv) > 1 and sys.argv[1] == 'gpu' else 'cpu'
SEEDS = (42, 7, 2026); ANCH = (288, 318, 348, 378); O = Path('artifacts/neural')
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
R = {}
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    w = anchor_weights(aid); last = lv[max(an)]
    lvl = np.array([lv[a].l for a in aid])
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    print(f'\n=== якорь {A} · обучающие {an} · строк {len(y):,} ===', flush=True)
    zs = []
    for s in SEEDS:
        t0 = time.perf_counter()
        zs.append(np.log1p(DirectXGB(config=XGBConfig(seed=s, device=DEV)).fit(
            X, y, feature_names=feats, sample_weight=w, z_offset=lvl
            ).predict(Xva, level=last.l)))
        print(f'  сид {s} за {time.perf_counter()-t0:.0f}с', flush=True)
    o = np.load(O/f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(),
                      **{f's{i}': v for i, v in enumerate(zs)}})
    b = pl.DataFrame({'user_id': o['user_id'], 'base': o['p0']*o['m0'],
                      'z': np.log1p(o['y'])}).join(t, on='user_id', how='inner')
    R[A] = dict(z=b['z'].to_numpy(), base=b['base'].to_numpy(),
                S=np.column_stack([b[f's{i}'].to_numpy() for i in range(len(SEEDS))]))
    del X, Xva; gc.collect()

def report(nseed):
    C, D = {}, {}
    print(f'\n{"="*58}\nсидов {nseed}\n{"="*58}')
    print(f'{"якорь":>7}{"база":>9}{"XGB":>9}{"C":>10}{"D":>9}{"alpha":>9}')
    for A in ANCH:
        r = R[A]; p = r['S'][:, :nseed].mean(1)
        e = r['z'] - r['base']; dd = p - r['base']; dd = dd - dd.mean()
        ec = e - e.mean(); C[A], D[A] = float((ec*dd).mean()), float((dd*dd).mean())
        print(f'{A:>7}{e.std():>9.5f}{(r["z"]-p).std():>9.5f}'
              f'{C[A]:>+10.5f}{D[A]:>9.5f}{C[A]/D[A]:>+9.4f}')
    tr = ANCH[:-1]; lam = sum(C[a] for a in tr)/sum(D[a] for a in tr)
    r = R[378]; p = r['S'][:, :nseed].mean(1)
    dd = p - r['base']; dd = dd - dd.mean(); e = r['z'] - r['base']
    s0, s1 = e.std(), (r['z'] - (r['base'] + lam*dd)).std()
    orc = s0 - np.sqrt(s0*s0 - C[378]**2/D[378])
    print(f'\n  lam по {list(tr)} = {lam:+.5f} (заморожен)')
    print(f'  ПЕРЕНОС на 378: {s0:.5f} → {s1:.5f}  {s0-s1:+.5f}  (оракул {orc:+.5f})')

for n in (1, 2, 3): report(n)
np.savez_compressed(O/'xgb_seeds.npz',
                    **{f'{k}_{A}': R[A][k] for A in ANCH for k in ('z','base','S')})
print('\nготово', flush=True)
