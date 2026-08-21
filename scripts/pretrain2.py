"""Massive supervised pretraining: случайный срез на пользователя за эпоху.

Отличия от прежнего этапа A, все существенные:

1. Случайный срез. Каждую эпоху пользователь берётся на СВОЁМ срезе, в
   следующую — обычно на другом. За 20-30 эпох сеть видит миллионы разных
   состояний, но в одном батче нет десятков почти одинаковых копий одного
   пользователя. Прежняя схема брала все срезы разом и оттого страдала
   зависимостью соседних окон.

2. Цели факторизованы и раздельны. Голова частоты не видит контекста
   суммы, голова поиска не видит каталога. Без этого семь запросов —
   просто семь наборов параметров без разных обязанностей.

3. Канальное разложение. Тождество gmv = gmv_search + gmv_cat подтверждено
   аудитом, но процессы за двумя слагаемыми могут быть разными, а общая
   GRU их смешивает.

4. Цели перехода. Уровень пользователя деревья знают хорошо; нейросети
   полезнее отвечать «ускоряется или затухает», чем заново восстанавливать
   «этот пользователь в среднем большой». Плюс время до следующей покупки.

Учителя-GBDT здесь нет вовсе: для любого T с наблюдаемым будущим все цели
считаются напрямую.
"""
from __future__ import annotations
import argparse, gc, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TARGETS = {                      # (голова, имена целей, тип лосса)
    "freq":       (("n_ord7", "n_ord30", "n_buyday30"), "mse"),
    "amount":     (("z_pos30", "aov30"), "mse"),
    "intent":     (("c3", "c7", "c14", "c30"), "bce"),
    "activity":   (("n_act7", "n_act30"), "mse"),
    "search":     (("z_s7", "z_s30", "c_s30"), "mix"),
    "catalog":    (("z_c7", "z_c30", "c_c30"), "mix"),
    "transition": (("dz30", "dn30", "tau3", "tau7", "tau14", "tau30"), "mix"),
}


def build_targets(df, T, users):
    """23 цели из наблюдаемого будущего. Никакого z0."""
    import polars as pl
    lg = lambda c: np.log1p(c.astype("float64")).astype("float32")

    def agg(lo, hi, sfx):
        t = (df.filter(pl.col("d").is_between(lo, hi))
               .group_by("user_id").agg(
                   g=pl.col("gmv").sum(), gs=pl.col("gmv_search").sum(),
                   gc=pl.col("gmv_cat").sum(), n=pl.col("to_ord").sum(),
                   act=pl.len(), buy=(pl.col("gmv") > 0).sum(),
                   first_buy=pl.col("d").filter(pl.col("gmv") > 0).min()))
        return (pl.DataFrame({"user_id": users}).join(t, on="user_id", how="left")
                  .with_columns(pl.exclude(["user_id", "first_buy"]).fill_null(0)))

    f30, f14, f7, f3 = (agg(T + 1, T + h, h) for h in (30, 14, 7, 3))
    past = agg(T - 29, T, "p")
    g30 = f30["g"].to_numpy().astype("float64")
    n30 = f30["n"].to_numpy().astype("float64")
    fb = f30["first_buy"].to_numpy()
    tau = np.where(np.isnan(fb.astype("float64")), 10**4, fb - T)
    out = {
        "n_ord7": lg(f7["n"].to_numpy()), "n_ord30": lg(n30),
        "n_buyday30": lg(f30["buy"].to_numpy()),
        "z_pos30": np.where(g30 > 0, np.log1p(g30), 0.0).astype("float32"),
        "aov30": np.where(n30 > 0, np.log1p(g30 / np.maximum(n30, 1)), 0.0).astype("float32"),
        "c3": (f3["g"].to_numpy() > 0).astype("float32"),
        "c7": (f7["g"].to_numpy() > 0).astype("float32"),
        "c14": (f14["g"].to_numpy() > 0).astype("float32"),
        "c30": (g30 > 0).astype("float32"),
        "n_act7": lg(f7["act"].to_numpy()), "n_act30": lg(f30["act"].to_numpy()),
        "z_s7": lg(f7["gs"].to_numpy()), "z_s30": lg(f30["gs"].to_numpy()),
        "c_s30": (f30["gs"].to_numpy() > 0).astype("float32"),
        "z_c7": lg(f7["gc"].to_numpy()), "z_c30": lg(f30["gc"].to_numpy()),
        "c_c30": (f30["gc"].to_numpy() > 0).astype("float32"),
        "dz30": (np.log1p(g30) - np.log1p(past["g"].to_numpy().astype("float64"))).astype("float32"),
        "dn30": (lg(n30) - lg(past["n"].to_numpy())).astype("float32"),
    }
    for h in (3, 7, 14, 30):
        out[f"tau{h}"] = (tau <= h).astype("float32")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-anchor", type=int, required=True)
    ap.add_argument("--earliest", type=int, default=178)
    ap.add_argument("--step", type=int, default=14)
    ap.add_argument("--users", type=int, default=120000)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--max-history", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("artifacts/neural/pre2"))
    ap.add_argument("--tb", type=Path, default=None)
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import polars as pl, torch
    from torch import nn
    from ecup import load_panel, selected_users
    from ecup.gapgru import GapGRUConfig, make_model, pick_device
    from ecup.tokens import TOKEN_FEATURES, build_tokens

    Q = ("freq", "amount", "intent", "activity", "search", "catalog", "transition")
    cuts = list(range(a.earliest, a.limit_anchor - 30 + 1, a.step))
    assert cuts[-1] + 30 <= a.limit_anchor, "нарушен пуржинг"
    print(f"срезов {len(cuts)}: {cuts[0]}…{cuts[-1]}", flush=True)
    df = load_panel(); a.cache.mkdir(parents=True, exist_ok=True)
    gi, ai = TOKEN_FEATURES.index("gap"), TOKEN_FEATURES.index("age")
    dev = pick_device()

    files = []
    for c in cuts:
        f = a.cache / f"c{c}_{a.max_len}_{a.users}.npz"
        if not f.exists():
            t0 = time.perf_counter()
            u = selected_users(df, c).to_numpy()
            if a.users and len(u) > a.users:
                keep = (u.astype(np.uint64) * 2654435761 % 1000000) < \
                       int(a.users / len(u) * 1000000)
                u = u[keep]
            us = pl.Series("user_id", u)
            X, L = build_tokens(df, c, us, a.max_history, a.max_len)
            np.savez(f, X=X, L=L, uid=u, **build_targets(df, c, us))
            print(f"  срез {c}: {X.shape} · {time.perf_counter()-t0:.0f}с", flush=True)
            del X; gc.collect()
        files.append(f)

    cfg = GapGRUConfig(n_features=len(TOKEN_FEATURES) - 2, max_len=a.max_len,
                       queries=Q, lr=a.lr, batch_size=a.batch_size,
                       epochs=a.epochs, seed=a.seed)
    model = make_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"параметров {sum(p.numel() for p in model.parameters()):,}", flush=True)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(a.seed)
    BIN = {"c3","c7","c14","c30","c_s30","c_c30","tau3","tau7","tau14","tau30"}
    W = {"freq":0.3,"amount":0.5,"intent":0.3,"activity":0.2,
         "search":0.3,"catalog":0.3,"transition":0.4}
    scale = {}                       # нормировка на начальную величину лосса

    nb_total = sum(len(np.load(f)["L"]) for f in files) // len(files)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=a.epochs * (nb_total // a.batch_size + 2))
    tb = None
    if a.tb is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(a.tb / f"pre2_{a.out.stem}"))
        except ImportError:
            pass

    for ep in range(a.epochs):
        t0, tot, nb = time.perf_counter(), 0.0, 0
        # Каждую эпоху пользователь берётся на СВОЁМ случайном срезе.
        data = [np.load(f) for f in files]
        pick = rng.integers(0, len(files), size=max(len(d["L"]) for d in data))
        order = []
        for fi, d_ in enumerate(data):
            n = len(d_["L"])
            order += [(fi, i) for i in np.where(pick[:n] == fi)[0]]
        rng.shuffle(order)
        order = np.array(order, dtype=np.int64)
        for s in range(0, len(order), a.batch_size):
            ch = order[s:s + a.batch_size]
            loss = 0.0
            for fi in np.unique(ch[:, 0]):
                d_ = data[fi]; idx = np.sort(ch[ch[:, 0] == fi, 1])
                Xb = np.asarray(d_["X"][idx], dtype="float32"); Lb = d_["L"][idx]
                m = np.arange(a.max_len)[None, :] >= \
                    (a.max_len - np.minimum(Lb, a.max_len))[:, None]
                pr = np.zeros((len(idx), 3), dtype="float32")
                pr[:, 2] = np.log1p(Lb) - np.log1p(d_["L"]).mean()
                T_ = lambda v, t=torch.float32: torch.as_tensor(v, dtype=t, device=dev)
                _, _, fo = model(T_(np.delete(Xb, [gi, ai], axis=2)), T_(Xb[:, :, gi]),
                                 T_(Xb[:, :, ai]), T_(m, torch.bool), T_(pr),
                                 factor_out=True)
                for q, (names, _) in TARGETS.items():
                    pred = fo[q]
                    for j, nmt in enumerate(names):
                        y = T_(d_[nmt][idx] - (0.0 if nmt in BIN else d_[nmt].mean()))
                        l = bce(pred[:, j], y) if nmt in BIN else mse(pred[:, j], y)
                        if nmt not in scale:
                            scale[nmt] = max(float(l.item()), 1e-3)
                        loss = loss + W[q] * l / scale[nmt]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch + 1 < sched.total_steps:
                sched.step()
            tot += float(loss.item()); nb += 1
        if tb is not None:
            tb.add_scalar("loss/pretrain2", tot / max(nb, 1), ep + 1)
        print(f"  эпоха {ep+1}/{a.epochs}  loss {tot/max(nb,1):.5f}  "
              f"примеров {len(order):,}  {time.perf_counter()-t0:.0f}с", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "epoch": ep + 1, "queries": Q}, a.out)
        del data; gc.collect()
    print(f"сохранено {a.out}\nГОТОВО")


if __name__ == "__main__":
    main()
