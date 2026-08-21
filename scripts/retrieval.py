"""Cross-user retrieval: память наблюдавшихся исходов, без учителя-GBDT.

Ключевое упрощение: память не требует честного z_0 на каждом историческом
срезе. Достаточно наблюдаемого будущего:

    M = { (h_{u,T},  Z^30_{u,T},  C^30_{u,T},  N^ord_{u,T}) }

для любого T, где окно T+1..T+30 наблюдается. Это снимает ограничение,
из-за которого заглохли плотные остаточные срезы: там на каждый срез
нужен был OOF-учитель, а он требует трёх обучающих якорей.

Запрос q = h_{u,A}. Допустимы только состояния с T + 30 <= A — тот же
горизонтальный пуржинг. Тот же user_id из соседей исключается: проверяем
именно кросс-пользовательскую локальность, а не персональный эффект,
который уже закрыт отрицательной автокорреляцией.

То, что оценка соседей во многом продублирует деревья, роли не играет:
маржинальный критерий вычтет проекцию на найденные направления.
"""
from __future__ import annotations
import argparse, gc, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HORIZON = 30


def targets(df, T, uid):
    import polars as pl
    t = (df.filter(pl.col("d").is_between(T + 1, T + HORIZON))
           .group_by("user_id")
           .agg(g=pl.col("gmv").sum(), n=pl.col("to_ord").sum()))
    k = (pl.DataFrame({"user_id": uid}).join(t, on="user_id", how="left")
           .with_columns(pl.exclude("user_id").fill_null(0)))
    g = k["g"].to_numpy().astype("float64")
    return (np.log1p(g).astype("float32"), (g > 0).astype("float32"),
            np.log1p(k["n"].to_numpy()).astype("float32"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("artifacts/neural/gapgru_evt_ckpt_fold1.pt"))
    ap.add_argument("--tokens", type=Path, default=Path("artifacts/neural/tokens"))
    ap.add_argument("--dense", type=Path, default=Path("artifacts/neural/dense"))
    ap.add_argument("--query-anchor", type=int, default=378)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("artifacts/neural/retr"))
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import polars as pl, torch
    from ecup import load_panel
    from ecup.gapgru import GapGRUConfig, make_model, pick_device
    from ecup.tokens import TOKEN_FEATURES

    a.out.mkdir(parents=True, exist_ok=True)
    gi, ai = TOKEN_FEATURES.index("gap"), TOKEN_FEATURES.index("age")
    dev = pick_device(); df = load_panel()
    cfg = GapGRUConfig(n_features=len(TOKEN_FEATURES) - 2, max_len=192)
    model = make_model(cfg).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    model.eval()

    def embed(T, tok_dir):
        meta = np.load(tok_dir / f"meta_a{T}.npz")
        X = np.load(tok_dir / f"x_a{T}.npy", mmap_mode="r")
        uid, L = meta["user_id"], meta["lengths"]
        oof = tok_dir.parent / f"oof_a{T}.npz"
        oof = oof if oof.exists() else tok_dir / f"oof_a{T}.npz"
        if oof.exists():
            o = np.load(oof)
            common, ti, oi = np.intersect1d(uid, o["user_id"], return_indices=True)
            z0 = o["z0"][oi]; dis = o["z0_lgb"][oi] - o["z0_cb"][oi]
            uid, L = common, L[ti]; rows = ti
        else:
            z0 = np.zeros(len(uid), "float32"); dis = np.zeros(len(uid), "float32")
            rows = np.arange(len(uid))
        out = []
        with torch.no_grad():
            for s in range(0, len(uid), 4096):
                sl = slice(s, s + 4096)
                Xb = np.asarray(X[rows[sl]], dtype="float32"); Lb = L[sl]
                m = np.arange(192)[None, :] >= (192 - np.minimum(Lb, 192))[:, None]
                pr = np.stack([z0[sl] - z0.mean(), dis[sl],
                               np.log1p(Lb) - np.log1p(L).mean()], 1)
                T_ = lambda v, t=torch.float32: torch.as_tensor(v, dtype=t, device=dev)
                _, _, _, feat = model(T_(np.delete(Xb, [gi, ai], axis=2)),
                                      T_(Xb[:, :, gi]), T_(Xb[:, :, ai]),
                                      T_(m, torch.bool), T_(pr), return_features=True)
                out.append(feat.float().cpu().numpy())
        return uid, np.concatenate(out), z0

    A = a.query_anchor
    mem_T = sorted({int(f.stem.split("_a")[1])
                    for d_ in (a.tokens, a.dense) if d_.exists()
                    for f in d_.glob("meta_a*.npz")}
                   & {t_ for t_ in range(150, A + 1) if t_ + HORIZON <= A})
    print(f"срезов памяти: {len(mem_T)} · {mem_T}", flush=True)

    H, U, Z, C, N = [], [], [], [], []
    for T in mem_T:
        d_ = a.tokens if (a.tokens / f"meta_a{T}.npz").exists() else a.dense
        t0 = time.perf_counter()
        uid, h, _ = embed(T, d_)
        z, c, n = targets(df, T, pl.Series("user_id", uid))
        H.append(h); U.append(uid); Z.append(z); C.append(c); N.append(n)
        print(f"  срез {T}: {h.shape} · {time.perf_counter()-t0:.0f}с", flush=True)
        gc.collect()
    H = np.concatenate(H); U = np.concatenate(U)
    Z, C, N = np.concatenate(Z), np.concatenate(C), np.concatenate(N)
    print(f"память: {H.shape[0]:,} состояний, размерность {H.shape[1]}", flush=True)

    q_uid, q_h, q_z0 = embed(A, a.tokens)
    # Whitening + PCA: обучаются на памяти, применяются к запросам
    mu, sd = H.mean(0), H.std(0) + 1e-6
    Hs = (H - mu) / sd
    _, _, Vt = np.linalg.svd(Hs[np.random.default_rng(0).choice(len(Hs),
                             min(50000, len(Hs)), replace=False)], full_matrices=False)
    P = Vt[:a.dim].T
    Hp = (Hs @ P).astype("float32"); Qp = (((q_h - mu) / sd) @ P).astype("float32")
    np.savez_compressed(a.out / f"memory_a{A}.npz", Hp=Hp, U=U, Z=Z, C=C, N=N,
                        Qp=Qp, q_uid=q_uid, q_z0=q_z0)
    print(f"сохранено {a.out}/memory_a{A}.npz · запросов {len(q_uid):,}")
    print("ГОТОВО")


if __name__ == "__main__":
    main()
