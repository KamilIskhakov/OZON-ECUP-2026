"""Боевое направление v27_longmoney на якоре 408.

Блок беcкэповой истории, нормированной на длину наблюдения, дал на двух
якорях парно +0.00249 (348) и +0.00276 (378) — на порядок больше всего,
что найдено после годового фильтра. Рыночная часть (rel_*) на 348 дала
-0.00015, поэтому в боевую версию входит ТОЛЬКО блок life_*.

Механизм: max_history обрезает историю (на 378 при h=300 видно 296 дней
из 378, на 408 — 300 из 408), а сырые пожизненные суммы несопоставимы
между якорями с разной длиной наблюдения. Чтобы дерево само получило
отношение, ему нужно сперва расщепиться по hist_span, а затем внутри
каждой ветки использовать свои пороги — это умножает нужную глубину.
Восемь признаков «на день наблюдения» чинят и обрезку, и дрейф.

Направление считается для ОБОИХ семейств при их боевых весах 0.4/0.6:

    d = 0.4 (z_lgb_new - z_lgb_old) + 0.6 (z_cb_new - z_cb_old)

Демеанируется: уровень v23 задан ручкой a_m = 0.18 по лидерборду.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
from ecup.catboost_model import CatBoostConfig, HurdleCatBoost
from ecup.market import market_features, _market

O = Path('artifacts/neural'); HIST = (240, 300, 365); SEEDS = (42, 7); A_M = 0.18
W_LGB, W_CB = 0.4, 0.6
df = load_panel(); mkt = _market(df)
FIN = SplitConfig(max_history=300, with_state=True).final_anchor
print(f'боевой якорь {FIN}', flush=True)

acc = {k: [] for k in ('lgb_old', 'lgb_new', 'cb_old', 'cb_new')}
for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
    an = sp.refit_anchors()
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); uid_tr = Xd['user_id'].to_numpy(); del Xd; gc.collect()

    t0 = time.perf_counter(); nm = None
    NEW = None
    for a in sorted(set(aid)):
        m = aid == a
        B, nm = market_features(df, int(a), uid_tr[m], mkt)
        if NEW is None:
            keep = [i for i, c in enumerate(nm) if c.startswith('life_')]
            nm_l = [nm[i] for i in keep]
            NEW = np.zeros((len(y), len(keep)), dtype='float32')
        NEW[m] = B[:, keep]
    fin = build_anchor(df, FIN, sp, None, with_target=False)
    Xte, _ = to_matrix(fin.X, feats); uid = fin.X['user_id'].to_numpy()
    Bte, _ = market_features(df, FIN, uid, mkt)
    NEWte = Bte[:, keep]
    X2 = np.hstack([X, NEW]); Xte2 = np.hstack([Xte, NEWte]); f2 = feats + nm_l
    p_base = last.p_bar; m_off = last.l_plus + A_M
    print(f'\n=== история {h} · якорей {len(an)} · строк {len(y):,} · '
          f'блок {len(nm_l)} за {time.perf_counter()-t0:.0f}с ===', flush=True)

    for s in SEEDS:
        for fam in ('lgb', 'cb'):
            for tag, (Xt, Xv, ff) in (('old', (X, Xte, feats)), ('new', (X2, Xte2, f2))):
                t0 = time.perf_counter()
                if fam == 'lgb':
                    mc = ModelConfig(seed=s)
                    hm = HurdleGBDT(config=mc).fit(
                        Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci,
                        early_stopping_rounds=mc.early_stopping_rounds,
                        eval_frac=mc.eval_frac, refit_full=mc.refit_full)
                else:
                    mc = CatBoostConfig(seed=s)
                    hm = HurdleCatBoost(config=mc).fit(
                        Xt, y, feature_names=ff, sample_weight=w, z_offset=zo, clf_init=ci,
                        early_stopping_rounds=mc.early_stopping_rounds,
                        eval_frac=mc.eval_frac, refit_full=mc.refit_full)
                z_ = np.log1p(hm.predict(Xv, p_target=p_base, m_offset=m_off))
                acc[f'{fam}_{tag}'].append((uid, z_))
                print(f'  {fam} {tag} сид {s}: std z {z_.std():.4f} · '
                      f'{time.perf_counter()-t0:.0f}с', flush=True)
    del X, X2, Xte, Xte2; gc.collect()

ref = np.load(O / 'dz_prod_a408.npz')['user_id']
def align(lst):
    out = []
    for uid_, z_ in lst:
        t = pl.DataFrame({'user_id': uid_, 'z': z_})
        out.append(pl.DataFrame({'user_id': ref}).join(t, on='user_id', how='left')['z'].to_numpy())
    return np.mean(out, 0)

Z = {k: align(v) for k, v in acc.items()}
d_lgb = Z['lgb_new'] - Z['lgb_old']; d_cb = Z['cb_new'] - Z['cb_old']
d_tot = W_LGB * d_lgb + W_CB * d_cb
d = d_tot - d_tot.mean()
print(f'\nLGB       std {d_lgb.std():.5f} · среднее {d_lgb.mean():+.5f}')
print(f'CatBoost  std {d_cb.std():.5f} · среднее {d_cb.mean():+.5f}')
print(f'corr(LGB, CB) {np.corrcoef(d_lgb, d_cb)[0,1]:+.4f}')
print(f'итоговое направление: std {d.std():.5f}, среднее {d.mean():+.2e}')
np.savez_compressed(O / 'longmoney_prod_a408.npz', user_id=ref, d=d, d_tot=d_tot,
                    d_lgb=d_lgb, d_cb=d_cb, **{f'z_{k}': v for k, v in Z.items()})
print('готово', flush=True)
