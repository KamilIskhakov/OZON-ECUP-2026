"""Честный базовый прогноз на каждом обучающем якоре: leave-one-anchor-out.

Сеть учится объяснять ОСТАТОК сильной модели. Если z0 на якоре a получен
моделью, видевшей якорь a, остаток искусственно мал и содержит следы
подгонки, а не непокрытый сигнал: сеть выучит их, покажет прекрасную
валидацию и рассыплется. Поэтому для каждого якоря модель переобучается
на остальных пяти.

Уровень при предсказании берётся собственный для якоря a. Это сознательно:
уровень окна калибруется отдельно и точно, и остаток должен нести только
форму. Иначе сеть первым делом выучит межоконный сдвиг уровня — то есть
ровно то, что уже решено одним числом.
"""
import warnings, sys, time, gc, numpy as np, polars as pl
warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
from ecup import load_panel, SplitConfig, ModelConfig, HurdleGBDT, build_anchor, \
                 build_training_set, to_matrix, anchor_weights, hurdle_glue
from ecup.dataset import anchor_offsets
from ecup.catboost_model import HurdleCatBoost, CatBoostConfig
from pathlib import Path

OUT = Path('artifacts/neural'); OUT.mkdir(parents=True, exist_ok=True)
sp = SplitConfig(max_history=300, n_train_anchors=6, with_state=True)
an = sp.train_anchors(); df = load_panel()
Xtr_df, ytr, aid, lv = build_training_set(df, an, sp, None, verbose=False)
uid = Xtr_df['user_id'].to_numpy()
Xtr, feats = to_matrix(Xtr_df); del Xtr_df; gc.collect()
ci, zo = anchor_offsets(aid, lv)
print(f'якоря {an} · строк {len(ytr):,} · признаков {len(feats)}\n', flush=True)

def fit_pair(idx, seed=42):
    """LGB и CatBoost на подмножестве строк; лимиты выше найденных оптимумов."""
    out = []
    for cfg, cls, key, cap in ((ModelConfig(seed=seed), HurdleGBDT, 'n_estimators', 800),
                               (CatBoostConfig(seed=seed), HurdleCatBoost, 'iterations', 2000)):
        cfg.clf_params[key] = cap; cfg.reg_params[key] = cap
        out.append(cls(config=cfg).fit(
            Xtr[idx], ytr[idx], feature_names=feats,
            sample_weight=anchor_weights(aid[idx]), z_offset=zo[idx], clf_init=ci[idx],
            early_stopping_rounds=100))
    return out

for a in an:
    t = time.perf_counter()
    tr, te = np.where(aid != a)[0], np.where(aid == a)[0]
    lvl = lv[a]
    zs, ps, ms = [], [], []
    for m in fit_pair(tr):
        p, mm = m.predict_parts(Xtr[te], p_target=lvl.p_bar, m_offset=lvl.l_plus)
        mm = np.clip(mm, 0.0, None)
        zs.append(np.log1p(hurdle_glue(p, mm))); ps.append(p); ms.append(mm)
        del m; gc.collect()
    z0 = np.mean(zs, axis=0)
    z = np.log1p(ytr[te]); r = z - z0
    np.savez_compressed(OUT / f'oofpm_a{a}.npz', user_id=uid[te], y=ytr[te],
                        z0=z0.astype('float32'), z0_lgb=zs[0].astype('float32'),
                        z0_cb=zs[1].astype('float32'),
                        p0=np.mean(ps, 0).astype('float32'),
                        m0=np.mean(ms, 0).astype('float32'))
    print(f'якорь {a}: n {len(te):,} · shape {r.std():.5f} · bias {r.mean():+.5f} · '
          f'std(z0) {z0.std():.4f} · {time.perf_counter()-t:.0f}с', flush=True)

val = build_anchor(df, sp.val_anchor, sp, None)
Xva, _ = to_matrix(val.X, feats)
lvl = lv[max(an)]
zs, ps, ms = [], [], []
for m in fit_pair(np.arange(len(ytr))):
    p, mm = m.predict_parts(Xva, p_target=lvl.p_bar, m_offset=lvl.l_plus)
    mm = np.clip(mm, 0.0, None)
    zs.append(np.log1p(hurdle_glue(p, mm))); ps.append(p); ms.append(mm)
    del m; gc.collect()
z0 = np.mean(zs, axis=0); z = np.log1p(val.y)
np.savez_compressed(OUT / f'oofpm_a{sp.val_anchor}.npz',
                    user_id=val.X['user_id'].to_numpy(), y=val.y,
                    z0=z0.astype('float32'), z0_lgb=zs[0].astype('float32'),
                    z0_cb=zs[1].astype('float32'),
                    p0=np.mean(ps, 0).astype('float32'),
                    m0=np.mean(ms, 0).astype('float32'))
print(f'\nвалидационный якорь {sp.val_anchor}: shape {(z-z0).std():.5f} '
      f'(обучен на всех {len(an)} якорях)', flush=True)
print('ГОТОВО')
