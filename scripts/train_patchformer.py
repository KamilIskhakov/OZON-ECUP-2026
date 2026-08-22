"""Обучение многомасштабного трансформера на остатке.

Случайный срез на пользователя за эпоху — но только среди срезов
с ЧЕСТНЫМ z0: остаточная голова обучается исправлять деревья, и брать
in-sample прогноз означало бы вернуть утечку через учителя. Доступны
основные якоря плюс плотные срезы, собранные walk-forward.

Пуржинг: срез T допустим для фолда с оценкой на E, если T + 30 <= (E - 30),
то есть целевое окно среза не пересекается с окном якоря подбора alpha.
"""
from __future__ import annotations
import argparse, sys, time, gc
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FOLDS = [(318, 348), (348, 378)]          # (граница обучения, якорь оценки)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/neural/patchformer.json"))
    a = ap.parse_args()
    import warnings; warnings.filterwarnings("ignore")
    import polars as pl, torch, json
    from torch import nn
    from ecup import load_panel
    from ecup.patches import build_patches, add_shape, N_CH
    from ecup.patchformer import PatchFormerConfig, make_patchformer
    from ecup.gapgru import pick_device
    from ecup.directions import marginal_gain

    dev = pick_device(); df = load_panel()
    O = Path("artifacts/neural")
    src = {}
    for f in list(O.glob("oof_a*.npz")) + list((O/"dense").glob("oof_a*.npz")):
        src.setdefault(int(f.stem.split("_a")[1]), f)
    print(f"срезов с честным z0: {sorted(src)}", flush=True)

    Jann = np.arange(2, 11)
    wann = np.cos(2*np.pi*(-379.5+30*Jann.astype(float))/365); wann -= wann.mean()

    def load(T):
        o = np.load(src[T]); uid = o["user_id"]; u = pl.Series("user_id", uid)
        P = build_patches(df, T, uid)
        def w(lo, hi, col="gmv"):
            t=(df.filter(pl.col("d").is_between(lo,hi)&pl.col("user_id").is_in(u))
                 .group_by("user_id").agg(g=pl.col(col).sum()))
            r=(pl.DataFrame({"user_id":uid}).join(t,on="user_id",how="left")
                 .with_columns(pl.col("g").fill_null(0.0)).sort("user_id"))
            return np.log1p(r["g"].to_numpy())
        dann = np.column_stack([w(T-364+30*j, T-335+30*j) for j in Jann]) @ wann
        A_=w(T+1-365,T+30-365); R=w(T-29,T); Rl=w(T-29-365,T-365)
        dyoy = np.column_stack([np.ones(len(A_))]+[f-f.mean() for f in (A_,R-Rl,R)]) @ \
               np.array([0.0,0.0263,-0.0087,-0.0145])
        p0 = o["p0"] if "p0" in o else np.full(len(uid), 0.5, "float32")
        m0 = o["m0"] if "m0" in o else o["z0"]*2
        prior = np.stack([o["z0"]-o["z0"].mean(), p0, m0-m0.mean(),
                          dann/max(dann.std(),1e-6), dyoy/max(dyoy.std(),1e-6),
                          w(T-29,T), w(T-89,T)], 1).astype("float32")
        # вспомогательные цели из наблюдаемого будущего
        def fut(h, col="gmv"):
            t=(df.filter(pl.col("d").is_between(T+1,T+h)&pl.col("user_id").is_in(u))
                 .group_by("user_id").agg(g=pl.col(col).sum(), n=pl.col("to_ord").sum(),
                                          b=(pl.col("gmv")>0).sum()))
            r=(pl.DataFrame({"user_id":uid}).join(t,on="user_id",how="left")
                 .with_columns(pl.exclude("user_id").fill_null(0)).sort("user_id"))
            return r
        f30, f14, f7 = fut(30), fut(14), fut(7)
        g30 = f30["g"].to_numpy().astype("float64")
        aux = {"z7": np.log1p(f7["g"].to_numpy()), "z14": np.log1p(f14["g"].to_numpy()),
               "z30": np.log1p(g30), "c30": (g30>0).astype("float32"),
               "n_ord": np.log1p(f30["n"].to_numpy()), "n_buy": np.log1p(f30["b"].to_numpy()),
               "z_s": np.log1p(fut(30,"gmv_search")["g"].to_numpy()),
               "z_c": np.log1p(fut(30,"gmv_cat")["g"].to_numpy())}
        return dict(uid=uid, xw=add_shape(P["week"]), xm=add_shape(P["mon"]),
                    pw=P["pos_week"], pm=P["pos_mon"], prior=prior,
                    z0=o["z0"], z=np.log1p(o["y"]),
                    aux={k: v.astype("float32") for k, v in aux.items()},
                    dann=dann, dyoy=dyoy)

    cache, results = {}, []
    def get(T):
        if T not in cache:
            t0 = time.perf_counter(); cache[T] = load(T)
            print(f"  срез {T}: {len(cache[T]['uid']):,} · "
                  f"{time.perf_counter()-t0:.0f}с", flush=True)
        return cache[T]

    for fi, (bnd, ev) in enumerate(FOLDS):
        Ts = sorted(t for t in src if t + 30 <= bnd)
        print(f"\n=== фолд {fi}: срезы {Ts} · оценка {ev} ===", flush=True)
        D = [get(T) for T in Ts]; Ev = get(ev)
        cfg = PatchFormerConfig(n_ch=2*N_CH, epochs=a.epochs, lr=a.lr,
                                batch_size=a.batch_size, seed=a.seed)
        m = make_patchformer(cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        nb = sum(len(d["uid"]) for d in D)//len(D)//cfg.batch_size + 2
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                                  total_steps=cfg.epochs*nb)
        mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(cfg.seed)
        T_ = lambda v, t=torch.float32: torch.as_tensor(v, dtype=t, device=dev)
        PW = T_(D[0]["pw"])[None]; PM = T_(D[0]["pm"])[None]
        # соответствие пользователь -> срез: наборы разные, выбор по user_id
        uni = np.unique(np.concatenate([d["uid"] for d in D]))
        pos = np.full((len(uni), len(D)), -1, np.int32)
        for k, d in enumerate(D): pos[np.searchsorted(uni, d["uid"]), k] = np.arange(len(d["uid"]))
        nok = (pos >= 0).sum(1)
        for ep in range(cfg.epochs):
            t0, tot, cnt = time.perf_counter(), 0.0, 0
            ch = np.floor(rng.random(len(uni))*nok).astype(np.int64)
            valid = pos >= 0; sel = (np.cumsum(valid,1) == (ch[:,None]+1)) & valid
            fidx = sel.argmax(1); rows = pos[np.arange(len(uni)), fidx]
            keep = nok > 0
            order = np.stack([fidx[keep], rows[keep]],1)
            order = order[np.lexsort((order[:,1], order[:,0]))]
            blocks = []
            for k in range(len(D)):
                r = order[order[:,0]==k,1]
                blocks += [(k, r[s:s+cfg.batch_size]) for s in range(0,len(r),cfg.batch_size)]
            rng.shuffle(blocks)
            for k, idx in blocks:
                d = D[k]
                dz, aux = m(T_(d["xw"][idx]), T_(d["xm"][idx]), PW, PM, T_(d["prior"][idx]))
                z0 = T_(d["z0"][idx]); tgt = T_(d["z"][idx])
                loss = mse(z0 + dz, tgt)
                for nm, wgt in cfg.aux.items():
                    y = T_(d["aux"][nm][idx])
                    loss = loss + wgt*(bce(aux[nm], y) if nm == "c30"
                                       else mse(aux[nm], y - d["aux"][nm].mean()))
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                if sch.last_epoch+1 < sch.total_steps: sch.step()
                tot += float(loss.item()); cnt += 1
            if (ep+1) % 5 == 0 or ep == cfg.epochs-1:
                print(f"  эпоха {ep+1}/{cfg.epochs} loss {tot/max(cnt,1):.5f} "
                      f"{time.perf_counter()-t0:.0f}с", flush=True)
        m.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(Ev["uid"]), 4096):
                sl = slice(s, s+4096)
                out.append(m(T_(Ev["xw"][sl]), T_(Ev["xm"][sl]), PW, PM,
                             T_(Ev["prior"][sl]))[0].float().cpu().numpy())
        d_new = np.concatenate(out)
        e = Ev["z"] - Ev["z0"]
        ex = [Ev["dann"], Ev["dyoy"]]
        if ev == 378:
            ex.append(np.load(O/"dz_a378.npz")["dz"])
        r = marginal_gain(e, d_new, existing=ex)
        print(f"  alpha {r['alpha_signed']:+.5f} · сольно {r['gain_solo']:+.5f} · "
              f"маржинально {r['gain_marginal']:+.5f} · std {d_new.std():.4f}", flush=True)
        results.append({k: float(v) if not isinstance(v, list) else v
                        for k, v in r.items() if k != "corr_with_existing"})
        np.save(O/f"d_patchformer_{ev}.npy", d_new)
        del m; gc.collect()
    sg = [np.sign(x["alpha_signed"]) for x in results]
    print(f"\nзнаки {[int(s) for s in sg]} — "
          f"{'совпадают' if len(set(sg))==1 else 'РАЗНЫЕ'}")
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ГОТОВО")

if __name__ == "__main__":
    main()
