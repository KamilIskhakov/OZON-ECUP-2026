"""BTYD-v2: форма временного риска, а не только его величина.

В v1 было четыре числа, и E[N30] как линейная ось оказался пуст.
Интересна ФОРМА hazard-кривой: два пользователя с одинаковым ожидаемым
числом покупок за месяц могут иметь совершенно разную временную
структуру риска — у одного всё в первую неделю, у другого равномерно.
В 183 агрегатах этой формы нет.

Блок зафиксирован заранее, десять признаков:
    P_alive, E[N7], E[N14], E[N30], E[N60],
    E[N7]/E[N30], E[N60]/(2 E[N30]), 30/E[N30],
    AOV на ЗАКАЗ, E[N30]*AOV

Размерность приведена: после перехода частоты на to_ord знаменатель
усадки чека тоже должен быть числом заказов, а не покупочных дней —
в v1 было рассогласование.

Сравнение — v1 против v2 на ОДНИХ моделях и сидах, а не против базы:
иначе измерим полезность BTYD вообще, а не улучшение вероятностной
модели. Порог +2e-4 на обоих якорях.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from btyd import rft, fit, predict

O = Path('artifacts/neural'); SEEDS = (42, 7); HIST = 300; EPS = 1e-6
df = load_panel(); OUT = {}


def feats_v2(A, uid, par=None):
    x, tx, T = rft(A, uid)
    buy = T > 0
    if par is None:
        idx = np.flatnonzero(buy)
        sub = np.random.default_rng(0).choice(idx, min(60000, len(idx)), replace=False)
        par, _ = fit(x[sub], tx[sub], T[sub])
    pa = np.zeros(len(x)); N = {}
    pa[buy], _ = predict(par, x[buy], tx[buy], T[buy], t=30)
    for h in (7, 14, 30, 60):
        v = np.zeros(len(x)); _, v[buy] = predict(par, x[buy], tx[buy], T[buy], t=h)
        N[h] = v
    # AOV на ЗАКАЗ — размерность согласована с частотой
    w = (df.filter((pl.col('d') <= A) & (pl.col('to_ord') > 0)).group_by('user_id')
           .agg(s=pl.col('gmv').sum(), n=pl.col('to_ord').sum().cast(pl.Float64)))
    j = pl.DataFrame({'user_id': uid}).join(w, on='user_id', how='left')
    gs = j['s'].fill_null(0.0).to_numpy(); nn = j['n'].fill_null(0.0).to_numpy()
    glob = gs.sum() / max(nn.sum(), 1); K = 3.0
    aov = (gs + K * glob) / (nn + K)
    cols = [pa, N[7], N[14], N[30], N[60],
            N[7] / (N[30] + EPS), N[60] / (2 * N[30] + EPS),
            30.0 / (N[30] + EPS), np.log1p(aov), np.log1p(N[30] * aov)]
    return np.nan_to_num(np.column_stack(cols), posinf=0, neginf=0).astype('float32'), par


def feats_v1(A, uid, par=None):
    from btyd_feat import btyd_feats
    return btyd_feats(A, uid, par)


NM1 = ['btyd_p_alive', 'btyd_en30', 'btyd_aov', 'btyd_gmv30']
NM2 = ['b2_alive', 'b2_n7', 'b2_n14', 'b2_n30', 'b2_n60', 'b2_q_near',
       'b2_q_long', 'b2_tau', 'b2_aov', 'b2_gmv30']
for A in (348, 378):
    sp = SplitConfig(max_history=HIST, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()
    t0 = time.perf_counter(); B1 = B2 = None
    for a in sorted(set(aid)):
        m = aid == a
        b1, _ = feats_v1(int(a), uid_tr[m]); b2, _ = feats_v2(int(a), uid_tr[m])
        if B1 is None:
            B1 = np.zeros((len(y), b1.shape[1]), 'float32')
            B2 = np.zeros((len(y), b2.shape[1]), 'float32')
        B1[m] = b1; B2[m] = b2
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    uid = val.X['user_id'].to_numpy()
    v1, _ = feats_v1(A, uid); v2, _ = feats_v2(A, uid)
    z = np.log1p(val.y)
    V = {'v1': (np.hstack([X, B1]), np.hstack([Xva, v1]), feats + NM1),
         'v2': (np.hstack([X, B2]), np.hstack([Xva, v2]), feats + NM2)}
    print(f'\n=== ЯКОРЬ {A} · v1 {len(NM1)} признаков · v2 {len(NM2)} · '
          f'{time.perf_counter()-t0:.0f}с ===', flush=True)
    for tag, (Xt, Xv, ff) in V.items():
        vs = []
        for s in SEEDS:
            hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
                Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci)
            p, m_ = hm.predict_parts(Xv, p_target=last.p_bar, m_offset=last.l_plus)
            vs.append(float((z - np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))).std()))
        OUT[(A, tag)] = np.array(vs)
        print(f'  {tag:<4}{" ".join(f"{v:.5f}" for v in vs)} · среднее {np.mean(vs):.5f}',
              flush=True)
    a1, a2 = OUT[(A, 'v1')], OUT[(A, 'v2')]
    print(f'  v2 - v1: {a1.mean()-a2.mean():+.5f} · парно '
          f'{" ".join(f"{v:+.5f}" for v in a1-a2)}', flush=True)
    del X, Xva; gc.collect()
print(f'\n{"якорь":>8}{"v1":>11}{"v2":>11}{"Δ":>11}')
for A in (348, 378):
    a1, a2 = OUT[(A, 'v1')], OUT[(A, 'v2')]
    print(f'{A:>8}{a1.mean():>11.5f}{a2.mean():>11.5f}{a1.mean()-a2.mean():>+11.5f}')
print('\nпорог +2e-4 на ОБОИХ якорях')
