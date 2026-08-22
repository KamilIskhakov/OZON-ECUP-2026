"""XGBoost третьим семейством: walk-forward на 288/318/348/378.

Проверяется не oracle-alpha на каждом якоре, а ПЕРЕНОС коэффициента:
lam оценивается только по 288/318/348 и без изменений применяется к 378.
Именно этого не хватило TabM, где oracle-alpha совпали на двух фолдах,
а public дал обратный знак.

Один сид, никакого HPO. База на каждом якоре — сохранённая OOF-пара
p0*m0, одинаковая по конструкции на всех четырёх якорях.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, load_panel, build_anchor, build_training_set,
                  to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.xgb_model import XGBConfig, HurdleXGB, DirectXGB

DEV = 'cuda' if len(sys.argv) > 1 and sys.argv[1] == 'gpu' else 'cpu'
ANCH = (288, 318, 348, 378); O = Path('artifacts/neural')
df = load_panel(); sp = SplitConfig(max_history=300, with_state=True)
R = {}
for A in ANCH:
    an = [a for a in sp.train_anchors() if a + 30 <= A]     # только законные
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    lvl = np.array([lv[a].l for a in aid])
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    print(f'\n=== якорь {A} · обучающие {an} · строк {len(y):,} ===', flush=True)

    cfg = XGBConfig(seed=42, device=DEV)
    t0 = time.perf_counter()
    zh = np.log1p(HurdleXGB(config=cfg).fit(
        X, y, feature_names=feats, sample_weight=w, z_offset=zo, clf_init=ci
        ).predict(Xva, p_target=last.p_bar, m_offset=last.l_plus))
    print(f'  hurdle обучен за {time.perf_counter()-t0:.0f}с', flush=True)
    t0 = time.perf_counter()
    zd = np.log1p(DirectXGB(config=cfg).fit(
        X, y, feature_names=feats, sample_weight=w, z_offset=lvl
        ).predict(Xva, level=last.l))
    print(f'  direct обучен за {time.perf_counter()-t0:.0f}с', flush=True)

    # выравнивание с сохранённой базой по user_id
    o = np.load(O/f'oofpm_a{A}.npz')
    t = pl.DataFrame({'user_id': val.X['user_id'].to_numpy(), 'h': zh, 'd': zd})
    b = pl.DataFrame({'user_id': o['user_id'], 'base': o['p0']*o['m0'],
                      'z': np.log1p(o['y'])}).join(t, on='user_id', how='inner')
    assert len(b) == len(o['user_id']), f'{len(b)} против {len(o["user_id"])}'
    z, base = b['z'].to_numpy(), b['base'].to_numpy(); e = z - base
    R[A] = dict(z=z, base=base, e=e, h=b['h'].to_numpy(), d=b['d'].to_numpy())
    print(f'  база {e.std():.5f} · hurdle {(z-R[A]["h"]).std():.5f} · '
          f'direct {(z-R[A]["d"]).std():.5f}', flush=True)
    del X, Xva; gc.collect()

for tag in ('h', 'd'):
    print(f'\n{"="*66}\n{"HURDLE" if tag=="h" else "DIRECT"}-XGB\n{"="*66}')
    print(f'{"якорь":>7}{"база":>9}{"XGB":>9}{"rho":>9}{"C":>10}{"D":>9}'
          f'{"alpha":>9}{"смесь 0.1":>11}')
    C, D = {}, {}
    for A in ANCH:
        r = R[A]; dd = r[tag] - r['base']; dd = dd - dd.mean()
        eA = r['e'] - r['e'].mean()
        C[A], D[A] = float((eA*dd).mean()), float((dd*dd).mean())
        rho = float(np.corrcoef(r['e'], r['z']-r[tag])[0,1])
        s01 = (r['z'] - (0.9*r['base'] + 0.1*r[tag])).std()
        print(f'{A:>7}{r["e"].std():>9.5f}{(r["z"]-r[tag]).std():>9.5f}{rho:>+9.4f}'
              f'{C[A]:>+10.5f}{D[A]:>9.5f}{C[A]/D[A]:>+9.4f}{s01:>11.5f}')
    tr = ANCH[:-1]; lam = sum(C[a] for a in tr)/sum(D[a] for a in tr)
    print(f'\n  lam по {list(tr)} = {lam:+.5f}  (заморожен)')
    r = R[378]; dd = r[tag] - r['base']; dd = dd - dd.mean()
    s0, s1 = r['e'].std(), (r['z'] - (r['base'] + lam*dd)).std()
    print(f'  ПЕРЕНОС на 378: {s0:.5f} → {s1:.5f}  {s0-s1:+.5f}')
    print(f'  (oracle на самом 378 дал бы {s0-np.sqrt(s0*s0-C[378]**2/D[378]):+.5f})')
np.savez_compressed(O/'xgb_walkforward.npz',
                    **{f'{k}_{A}': R[A][k] for A in ANCH for k in ('z','base','h','d')})
print('\nготово', flush=True)
