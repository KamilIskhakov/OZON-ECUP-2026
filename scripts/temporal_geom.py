"""Ресурс временной ранней остановки ПОВЕРХ v28.

Standalone политика A даёт +0.00022 на одиночной LGB. Но сегодня
дважды выяснилось, что прирост над одиночной моделью не является
оценкой боевого выигрыша: у life было +0.0025 standalone и +0.00016
поверх v23, у depth +0.00060 на LGB-ансамбле и +0.00001 поверх v28.

Здесь считается ансамблевое направление в боевой геометрии:

    d_temp = 0.4 (z_LGB^{временная ES} - z_LGB^{случайная ES})

CatBoost не трогаем: для него временная схема не валидирована.

База — точный v28 = z_v23 + 0.24 (d_life - mean d_life).

В отличие от life и depth, эта конструкция по построению НЕ должна
коррелировать с годовым фильтром: она меняет не информацию, а число
деревьев. Диагностические корреляции это проверят.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import (SplitConfig, ModelConfig, load_panel, build_anchor,
                  build_training_set, to_matrix, anchor_weights, hurdle_glue)
from ecup.dataset import anchor_offsets
from ecup.model import HurdleGBDT
import lightgbm as lgb

A = 378; O = Path('artifacts/neural'); HIST = (240, 300, 365); SEEDS = (42, 7)
BIG = 3000; HL = 90.0
df = load_panel(); acc = {'old': [], 'new': []}

for h in HIST:
    sp = SplitConfig(max_history=h, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= A]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    ci, zo = anchor_offsets(aid, lv); last = lv[max(an)]
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    val = build_anchor(df, A, sp, None); Xva, _ = to_matrix(val.X, feats)
    uid = val.X['user_id'].to_numpy(); es_a = max(an)
    pos = y > 0; z = np.log1p(y) - zo
    wts = lambda a_: np.power(0.5, (a_.max() - a_).astype('float64') / HL)
    print(f'\n=== история {h} · ES на {es_a} ===', flush=True)
    for s in SEEDS:
        t0 = time.perf_counter()
        hm = HurdleGBDT(config=ModelConfig(seed=s)).fit(
            X, y, feature_names=feats, sample_weight=anchor_weights(aid),
            z_offset=zo, clf_init=ci)
        p, m_ = hm.predict_parts(Xva, p_target=last.p_bar, m_offset=last.l_plus)
        acc['old'].append((uid, np.log1p(hurdle_glue(p, np.clip(m_, 0, None)))))
        mc = ModelConfig(seed=s)
        cp = {**mc.clf_params, 'n_estimators': BIG, 'verbose': -1, 'n_jobs': -1}
        rp = {**mc.reg_params, 'n_estimators': BIG, 'verbose': -1, 'n_jobs': -1}
        tr = aid != es_a; es = aid == es_a
        c0 = lgb.LGBMClassifier(random_state=s, **cp).fit(
            X[tr], pos[tr].astype(np.int8), sample_weight=wts(aid[tr]), init_score=ci[tr],
            eval_set=[(X[es], pos[es].astype(np.int8))], eval_init_score=[ci[es]],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(100, verbose=False)])
        pt = tr & pos; pe = es & pos
        r0 = lgb.LGBMRegressor(random_state=s, **rp).fit(
            X[pt], z[pt], sample_weight=wts(aid[pt]),
            eval_set=[(X[pe], z[pe])], eval_metric='l2',
            callbacks=[lgb.early_stopping(100, verbose=False)])
        kp = c0.best_iteration_ or BIG; km = r0.best_iteration_ or BIG
        clf = lgb.LGBMClassifier(random_state=s, **{**cp, 'n_estimators': kp}).fit(
            X, pos.astype(np.int8), sample_weight=wts(aid), init_score=ci)
        reg = lgb.LGBMRegressor(random_state=s, **{**rp, 'n_estimators': km}).fit(
            X[pos], z[pos], sample_weight=wts(aid[pos]))
        raw = clf.predict(Xva, raw_score=True)
        pp = 1 / (1 + np.exp(-(raw + np.log(last.p_bar / (1 - last.p_bar)))))
        mm = reg.predict(Xva) + last.l_plus
        acc['new'].append((uid, np.log1p(hurdle_glue(pp, np.clip(mm, 0, None)))))
        print(f'  сид {s}: деревьев старая {hm.best_iters} · новая ({kp}, {km}) · '
              f'{time.perf_counter()-t0:.0f}с', flush=True)
    del X, Xva; gc.collect()

G = np.load(O / f'life_geom_a{A}.npz')
ref = G['user_id']; z_v23 = G['z_v23']; d_life = G['d_life']
o = np.load(O / f'oofpm_a{A}.npz')
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(pl.DataFrame({'user_id': u_, 'p': v_}),
                             on='user_id', how='left')['p'].to_numpy()
zt = key.join(pl.DataFrame({'user_id': o['user_id'], 'z': np.log1p(o['y'])}),
              on='user_id', how='left')['z'].to_numpy()
Z = {k: np.mean([al(*v) for v in lst], 0) for k, lst in acc.items()}
d_temp = 0.4 * (Z['new'] - Z['old'])
z28 = z_v23 + 0.24 * (d_life - d_life.mean())
e = zt - z28
print(f'\nLGB-ансамбль: старая ES {float((zt-Z["old"]).std()):.5f} · '
      f'новая {float((zt-Z["new"]).std()):.5f} · '
      f'Δ {float((zt-Z["old"]).std())-float((zt-Z["new"]).std()):+.5f}')
print(f'shape v28 {float(e.std()):.5f}')
C = float(((e - e.mean()) * (d_temp - d_temp.mean())).mean())
D = float(((d_temp - d_temp.mean())**2).mean())
a = C / max(D, 1e-15); dM = C * C / max(D, 1e-15)
new = np.sqrt(max(e.var() - dM, 0))
print(f'\nd_temp поверх v28: std {d_temp.std():.5f} · C {C:+.3e} · D {D:.3e}')
print(f'  alpha* {a:+.4f} · shape {float(e.std()):.5f} -> {new:.5f} '
      f'({float(e.std())-new:+.5f})')
al1 = float((e - (d_temp - d_temp.mean())).std())
print(f'  ПРИ alpha = 1: {al1:.5f} ({float(e.std())-al1:+.5f})')
dann = np.load(f'/tmp/d_annual_{A}.npy'); d16 = np.load(O / f'dz_a{A}.npz')['dz']
print(f'\nдиагностика: corr(temp, life) {np.corrcoef(d_temp, d_life)[0,1]:+.4f} · '
      f'corr(temp, annual) {np.corrcoef(d_temp, dann)[0,1]:+.4f} · '
      f'corr(temp, GRU) {np.corrcoef(d_temp, d16)[0,1]:+.4f}')
np.savez_compressed(O / f'temp_geom_a{A}.npz', user_id=ref, d_temp=d_temp, e28=e)
print('\nготово', flush=True)
