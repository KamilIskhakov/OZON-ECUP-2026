"""Плотные честные базовые прогнозы: остаточных примеров в разы больше.

Сейчас этап B учится на 4-5 якорях. Замер плотности показал, что плотная
временная разметка несёт информацию (stride 30 -> 10 дал GBDT +0.0005),
причём для GBDT почти вся она съедается зависимостью соседних окон,
а энкодер использует срезы иначе — как примеры перехода состояния.

Схема walk-forward: для среза T база строится моделью, обученной ТОЛЬКО
на якорях с концом целевого окна не позже T, то есть a + 30 <= T. Это тот
же горизонтальный пуржинг, на котором уже была поймана утечка в 0.005.

Считается на CPU и не конкурирует с обучением сети на GPU.
"""
from __future__ import annotations
import argparse, gc, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HORIZON = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=258)
    ap.add_argument("--stop", type=int, default=348)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--max-history", type=int, default=300)
    ap.add_argument("--n-anchors", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--out", type=Path, default=Path("artifacts/neural/dense"))
    ap.add_argument("--tokens", action="store_true", help="также собрать токены")
    a = ap.parse_args()

    import polars as pl
    from ecup import (ModelConfig, SplitConfig, HurdleGBDT, load_panel, build_anchor,
                      build_training_set, to_matrix, anchor_weights, hurdle_glue,
                      selected_users)
    from ecup.dataset import anchor_offsets
    from ecup.catboost_model import HurdleCatBoost, CatBoostConfig
    from ecup.tokens import build_tokens

    a.out.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    cuts = list(range(a.start, a.stop + 1, a.step))
    print(f"срезов {len(cuts)}: {cuts}", flush=True)

    for T in cuts:
        f = a.out / f"oof_a{T}.npz"
        if f.exists():
            print(f"  срез {T}: уже есть", flush=True); continue
        t0 = time.perf_counter()
        # якоря строго до горизонта: a + 30 <= T
        an = [T - HORIZON - 30 * k for k in range(a.n_anchors)]
        an = sorted(x for x in an if x >= 178)
        if len(an) < 3:
            print(f"  срез {T}: якорей {len(an)}, пропуск", flush=True); continue
        assert max(an) + HORIZON <= T, "нарушен пуржинг"

        sp = SplitConfig(max_history=a.max_history, n_train_anchors=len(an),
                         with_state=True)
        Xtr_df, ytr, aid, lv = build_training_set(df, an, sp, None, verbose=False)
        Xtr, feats = to_matrix(Xtr_df); del Xtr_df; gc.collect()
        w = anchor_weights(aid); ci, zo = anchor_offsets(aid, lv)
        tgt = build_anchor(df, T, sp, None)
        Xt, _ = to_matrix(tgt.X, feats)
        lvl = lv[max(an)]

        zs = []
        for cfg, cls, key, cap in ((ModelConfig(seed=42), HurdleGBDT, "n_estimators", 800),
                                   (CatBoostConfig(seed=42), HurdleCatBoost, "iterations", 2000)):
            cfg.clf_params[key] = cap; cfg.reg_params[key] = cap
            m = cls(config=cfg).fit(Xtr, ytr, feature_names=feats, sample_weight=w,
                                    z_offset=zo, clf_init=ci, early_stopping_rounds=100)
            p, mm = m.predict_parts(Xt, p_target=lvl.p_bar, m_offset=lvl.l_plus)
            zs.append(np.log1p(hurdle_glue(p, np.clip(mm, 0.0, None))))
            del m; gc.collect()
        z0 = np.mean(zs, axis=0)
        z = np.log1p(tgt.y)
        np.savez_compressed(f, user_id=tgt.X["user_id"].to_numpy(), y=tgt.y,
                            z0=z0.astype("float32"), z0_lgb=zs[0].astype("float32"),
                            z0_cb=zs[1].astype("float32"))
        print(f"  срез {T}: якоря {an} · n {len(z):,} · shape {(z-z0).std():.5f} · "
              f"bias {(z-z0).mean():+.5f} · {time.perf_counter()-t0:.0f}с", flush=True)

        if a.tokens:
            tf = a.out / f"x_a{T}.npy"
            if not tf.exists():
                u = pl.Series("user_id", tgt.X["user_id"].to_numpy())
                X, L = build_tokens(df, T, u, a.max_history, a.max_len)
                np.save(tf, X)
                np.savez(a.out / f"meta_a{T}.npz", user_id=u.to_numpy(), lengths=L)
                del X; gc.collect()
        del Xtr, Xt, tgt, w, ci, zo; gc.collect()
    print("ГОТОВО")


if __name__ == "__main__":
    main()
