"""Этап A: многозадачный претрейн энкодера на плотной временной нарезке.

Зачем он нужен. Остаток сильного ансамбля — крайне шумная цель: его R²
по сырому ряду измерен как +0.0008. Учить на нём энкодер с нуля значит
подгонять шум. Претрейн даёт энкодеру выучить МОДЕЛЬ ПОВЕДЕНИЯ на
плотной сетке срезов, где целей много и они не зашумлены вычитанием
чужого прогноза.

Плотность обоснована замером: при правильном числе деревьев уплотнение
якорей с 30 до 10 дней дало GBDT +0.0005 — то есть плотная разметка
действительно несёт информацию, просто GBDT почти всю её теряет на
зависимости соседних окон. Энкодер использует срезы иначе: как примеры
перехода состояния, а не как дополнительные строки.

Пуржинг строгий: срез T допустим, только если T + H <= A, где A — самый
ранний якорь, участвующий в оценке фолда (якорь подбора α). Отступ равен
ГОРИЗОНТУ, а не шагу — на путанице этих двух величин уже была поймана
утечка в 0.005.
"""
from __future__ import annotations

import argparse, gc, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HORIZON = 30


def cutoffs_for(limit_anchor: int, earliest: int, step: int) -> list[int]:
    """Срезы, целевые окна которых заканчиваются не позже limit_anchor."""
    last = limit_anchor - HORIZON
    return list(range(earliest, last + 1, step))


def targets_at(df, cutoff: int, users) -> dict:
    import polars as pl
    out = {}
    for h, tag in ((7, "7"), (HORIZON, "30")):
        t = (df.filter(pl.col("d").is_between(cutoff + 1, cutoff + h))
               .group_by("user_id")
               .agg(gmv=pl.col("gmv").sum(),
                    n_buy=(pl.col("gmv") > 0).sum(),
                    n_ord=pl.col("to_ord").sum()))
        a = (pl.DataFrame({"user_id": users}).join(t, on="user_id", how="left")
               .with_columns(pl.exclude("user_id").fill_null(0)))
        y = a["gmv"].to_numpy().astype("float64")
        out[f"z{tag}"] = np.log1p(y).astype("float32")
        out[f"c{tag}"] = (y > 0).astype("float32")
        out[f"nbuy{tag}"] = np.log1p(a["n_buy"].to_numpy()).astype("float32")
        out[f"nord{tag}"] = np.log1p(a["n_ord"].to_numpy()).astype("float32")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-anchor", type=int, required=True,
                    help="якорь подбора α того фолда, под который готовится претрейн")
    ap.add_argument("--earliest", type=int, default=178)
    ap.add_argument("--step", type=int, default=14)
    ap.add_argument("--users", type=int, default=60000)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--max-history", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", action="store_true",
                    help="вторая шкала времени: токены покупочных циклов")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("artifacts/neural/pretrain"))
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import polars as pl, torch
    from torch import nn
    from ecup import SplitConfig, load_panel, selected_users
    from ecup.gapgru import GapGRUConfig, make_model, pick_device
    from ecup.tokens import TOKEN_FEATURES, build_tokens, cycle_tokens

    cuts = cutoffs_for(a.limit_anchor, a.earliest, a.step)
    print(f"срезов {len(cuts)}: {cuts[0]}…{cuts[-1]} шаг {a.step} · "
          f"последний целевой день {cuts[-1] + HORIZON} <= {a.limit_anchor}", flush=True)
    assert cuts[-1] + HORIZON <= a.limit_anchor, "нарушен пуржинг"

    df = load_panel()
    a.cache.mkdir(parents=True, exist_ok=True)
    gi, ai = TOKEN_FEATURES.index("gap"), TOKEN_FEATURES.index("age")
    dev = pick_device()

    store = []
    for c in cuts:
        tag = "cyc" if a.cycles else "evt"
        f = a.cache / f"cut_{c}_{a.max_len}_{a.users}_{tag}.npz"
        if not f.exists():
            t0 = time.perf_counter()
            u = selected_users(df, c).to_numpy()
            if a.users and len(u) > a.users:
                # детерминированная подвыборка: один и тот же пользователь
                # должен попадать в неё на всех срезах
                keep = (u.astype(np.uint64) * 2654435761 % 1000000) < \
                       int(a.users / len(u) * 1000000)
                u = u[keep]
            us = pl.Series("user_id", u)
            X, L = build_tokens(df, c, us, a.max_history, a.max_len)
            extra = {}
            if a.cycles:
                C, _ = cycle_tokens(df, c, us, a.max_history)
                extra["C"] = C.astype("float16")
            np.savez(f, X=X, L=L, **extra, **targets_at(df, c, us))
            print(f"  срез {c}: {X.shape} · {time.perf_counter()-t0:.0f}с", flush=True)
            del X; gc.collect()
        store.append(f)

    cfg = GapGRUConfig(n_features=len(TOKEN_FEATURES) - 2, max_len=a.max_len,
                       lr=a.lr, batch_size=a.batch_size, epochs=a.epochs,
                       seed=a.seed, use_cycles=a.cycles)
    model = make_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(a.seed)
    n_per = sum(len(np.load(f)["L"]) for f in store)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=a.epochs * (n_per // a.batch_size + len(store)))
    print(f"примеров {n_per:,} · параметров "
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    for ep in range(a.epochs):
        t0, tot, nb = time.perf_counter(), 0.0, 0
        for f in rng.permutation(store):
            d = np.load(f)
            X_all, L_all = d["X"], d["L"]
            # уровень среза вычитается: сеть должна учить форму, а не
            # сезонный уровень конкретного окна — он калибруется отдельно
            z30 = d["z30"] - d["z30"].mean()
            order = rng.permutation(len(L_all))
            for s in range(0, len(order), a.batch_size):
                idx = np.sort(order[s:s + a.batch_size])
                Xb = np.asarray(X_all[idx], dtype="float32")
                L = L_all[idx]
                m = np.arange(a.max_len)[None, :] >= \
                    (a.max_len - np.minimum(L, a.max_len))[:, None]
                T = lambda v, t=torch.float32: torch.as_tensor(v, dtype=t, device=dev)
                prior = np.zeros((len(idx), 3), dtype="float32")
                prior[:, 2] = np.log1p(L) - np.log1p(L_all).mean()
                opt.zero_grad(set_to_none=True)
                cy = T(np.asarray(d["C"][idx], dtype="float32")) if a.cycles else None
                _, aux = model(T(np.delete(Xb, [gi, ai], axis=2)), T(Xb[:, :, gi]),
                               T(Xb[:, :, ai]), T(m, torch.bool), T(prior),
                               cycles=cy)
                loss = (mse(aux["z_abs"], T(z30[idx]))
                        + 0.3 * bce(aux["p"], T(d["c30"][idx]))
                        + 0.2 * mse(aux["n_buy"], T(d["nbuy30"][idx]))
                        + 0.1 * mse(aux["n_ord"], T(d["nord30"][idx])))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += float(loss.item()); nb += 1
            del d, X_all; gc.collect()
        print(f"  эпоха {ep+1}/{a.epochs}  loss {tot/max(nb,1):.5f}  "
              f"{time.perf_counter()-t0:.0f}с", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": ep + 1},
                   a.out)          # чекпоинт каждую эпоху: spot-инстанс прерывается штатно
    print(f"сохранено {a.out}\nГОТОВО")


if __name__ == "__main__":
    main()
