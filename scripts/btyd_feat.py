"""BTYD как признаки к GBDT, парный контроль.

Как направления выходы BG/NBD пусты (corr 0.003-0.017, знак не
переносится). Но дерево может использовать P(alive) нелинейно там,
где линейная проекция ничего не даёт. Четыре признака:
P_alive, E[N30], log AOV, log E[GMV30].
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
sys.path.insert(0, 'scripts')
from btyd import rft, fit, predict

O = Path('artifacts/neural'); SEEDS = (42, 7); HIST = 300
df = load_panel(); OUT = {}


def btyd_feats(A, uid, par=None):
    x, tx, T = rft(A, uid)
    buy = T > 0
    if par is None:
        idx = np.flatnonzero(buy)
        sub = np.random.default_rng(0).choice(idx, min(60000, len(idx)), replace=False)
        par, _ = fit(x[sub], tx[sub], T[sub])
    pa = np.zeros(len(x)); en = np.zeros(len(x))
    pa[buy], en[buy] = predict(par, x[buy], tx[buy], T[buy])
    w = (df.filter((pl.col('d') <= A) & (pl.col('gmv') > 0)).group_by('user_id')
           .agg(s=pl.col('gmv').sum(), n=pl.col('d').n_unique()))
    j = pl.DataFrame({'user_id': uid}).join(w, on='user_id', how='left')
    gs = j['s'].fill_null(0.0).to_numpy(); nn = j['n'].fill_null(0).to_numpy()
    glob = gs.sum() / max(nn.sum(), 1); K = 3.0
    aov = (gs + K * glob) / (nn + K)
    return np.column_stack([pa, en, np.log1p(aov), np.log1p(en * aov)]).astype('float32'), par



if __name__ == '__main__':
    for A in (348, 378):
        sp = SplitConfig(max_history=HIST, with_state=True)
        an = [a for a in sp.train_anchors() if a + 30 <= A]
        Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
        w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
        X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
        t0 = time.perf_counter(); NEW = None; par = None
        for a in sorted(set(aid)):
            m = aid == a
            B, par = btyd_feats(int(a), uid_tr[m])
            if NEW is None: NEW = np.zeros((len(y), B.shape[1]), dtype='float32')
            NEW[m] = B
        val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
        Bva, _ = btyd_feats(A, val.X['user_id'].to_numpy())
        z = np.log1p(val.y)
        nm = ['btyd_p_alive', 'btyd_en30', 'btyd_aov', 'btyd_gmv30']
        X2 = np.hstack([X, NEW]); Xva2 = np.hstack([Xva, Bva]); f2 = feats + nm
        print(f'\n=== ЯКОРЬ {A} · признаков {len(nm)} за {time.perf_counter()-t0:.0f}с ===',
              flush=True)
        for tag, (Xt, Xv, ff) in (('база', (X, Xva, feats)), ('+btyd', (X2, Xva2, f2))):
            vs = []
            for s in SEEDS:
                hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
                    Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
                p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
                vs.append(float((z - np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))).std()))
            OUT[(A, tag)] = np.array(vs)
            print(f'  {tag:<7}{" ".join(f"{v:.5f}" for v in vs)} · среднее {np.mean(vs):.5f}',
                  flush=True)
        b, n = OUT[(A, 'база')], OUT[(A, '+btyd')]
        print(f'  Δ {b.mean()-n.mean():+.5f} · парно {" ".join(f"{v:+.5f}" for v in b-n)}',
              flush=True)
        del X, X2, Xva, Xva2; gc.collect()
    print(f'\n{"якорь":>8}{"база":>11}{"+btyd":>11}{"Δ":>11}')
    for A in (348, 378):
        b, n = OUT[(A, 'база')], OUT[(A, '+btyd')]
        print(f'{A:>8}{b.mean():>11.5f}{n.mean():>11.5f}{b.mean()-n.mean():>+11.5f}')
