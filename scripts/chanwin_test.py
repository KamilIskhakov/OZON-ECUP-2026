"""Парный тест канального блока: одиночная модель, два якоря, два сида."""
import sys, warnings, gc, time, os; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from chanwin import channel_windows

SEEDS = (42, 7); HIST = 300
df = load_panel(); OUT = {}
for A in (348, 378):
    sp = SplitConfig(max_history=HIST, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
    t0 = time.perf_counter(); NEW = None
    for a in sorted(set(aid)):
        m = aid == a
        B, nm = channel_windows(df, int(a), uid_tr[m])
        if NEW is None: NEW = np.zeros((len(y), B.shape[1]), dtype='float32')
        NEW[m] = B
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Bva, _ = channel_windows(df, A, val.X['user_id'].to_numpy())
    z = np.log1p(val.y)
    X2 = np.hstack([X, NEW]); Xva2 = np.hstack([Xva, Bva]); f2 = feats + nm
    print(f'\n=== ЯКОРЬ {A} · базовых {X.shape[1]} · новых {len(nm)} за '
          f'{time.perf_counter()-t0:.0f}с ===', flush=True)
    for tag, (Xt, Xv, ff) in (('база', (X, Xva, feats)), ('+канал', (X2, Xva2, f2))):
        vs = []
        for s in SEEDS:
            t0 = time.perf_counter()
            hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
                Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
            p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
            vs.append(float((z - np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))).std()))
            print(f'  {tag:<8} сид {s}: {vs[-1]:.5f} · деревьев {hm.best_iters} · '
                  f'{time.perf_counter()-t0:.0f}с', flush=True)
        OUT[(A, tag)] = np.array(vs)
    b, n = OUT[(A, 'база')], OUT[(A, '+канал')]
    print(f'  Δ {b.mean()-n.mean():+.5f} · парно '
          f'{" ".join(f"{x:+.5f}" for x in b-n)}', flush=True)
    del X, X2, Xva, Xva2; gc.collect()
print(f'\n{"якорь":>8}{"база":>11}{"+канал":>11}{"Δ":>11}')
for A in (348, 378):
    b, n = OUT[(A, 'база')], OUT[(A, '+канал')]
    print(f'{A:>8}{b.mean():>11.5f}{n.mean():>11.5f}{b.mean()-n.mean():>+11.5f}')
print('\nготово', flush=True)
