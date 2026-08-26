"""Парный тест воронковых параметров, два сида, два якоря."""
import sys, warnings, gc, time; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
warnings.filterwarnings('ignore')
import numpy as np
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from funnel import funnel_features

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
        B, nm = funnel_features(df, int(a), uid_tr[m])
        if NEW is None: NEW = np.zeros((len(y), B.shape[1]), 'float32')
        NEW[m] = B
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    Bva, _ = funnel_features(df, A, val.X['user_id'].to_numpy())
    z = np.log1p(val.y)
    X2 = np.hstack([X, NEW]); Xva2 = np.hstack([Xva, Bva]); f2 = feats + nm
    print(f'\n=== ЯКОРЬ {A} · новых {len(nm)} за {time.perf_counter()-t0:.0f}с ===',
          flush=True)
    for tag, (Xt, Xv, ff) in (('база', (X, Xva, feats)), ('+воронка', (X2, Xva2, f2))):
        vs = []
        for s in SEEDS:
            hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
                Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
            p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
            vs.append(float((z - np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))).std()))
        OUT[(A, tag)] = np.array(vs)
        print(f'  {tag:<10}{" ".join(f"{v:.5f}" for v in vs)} · среднее {np.mean(vs):.5f}',
              flush=True)
    b, n = OUT[(A, 'база')], OUT[(A, '+воронка')]
    print(f'  Δ {b.mean()-n.mean():+.5f} · парно {" ".join(f"{v:+.5f}" for v in b-n)}',
          flush=True)
    del X, X2, Xva, Xva2; gc.collect()
print(f'\n{"якорь":>8}{"база":>11}{"+воронка":>11}{"Δ":>11}')
for A in (348, 378):
    b, n = OUT[(A, 'база')], OUT[(A, '+воронка')]
    print(f'{A:>8}{b.mean():>11.5f}{n.mean():>11.5f}{b.mean()-n.mean():>+11.5f}')
print('\nшлюз +2e-4 на ОБОИХ якорях, задан до просмотра')
