"""Gap-GRU на остатке ансамбля: два временных фолда, критерий переносимости.

Критерий успеха задан заранее и не подлежит пересмотру по факту: поправка
принимается, только если z0 + α·Δz бьёт z0 на ОБОИХ фолдах. Плюс на одном
и минус на другом означает шум, а не сигнал — на нём мы уже один раз
обожглись, приняв за находку разницу 0.00003.

Фолды сдвинуты по времени и не пересекаются целевыми окнами:

    фолд 1: обучение 198…288 · подбор α на 318 · оценка на 348
    фолд 2: обучение 198…318 · подбор α на 348 · оценка на 378

α подбирается ОТДЕЛЬНО от сети и на отдельном якоре — той же параболой
из двух точек, что закрыла долю CatBoost. Учить α вместе с zero-init
головой бессмысленно: произведение α·Δz вырождено, сеть просто
отмасштабирует Δz.
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TRAIN_ANCHORS = [198, 228, 258, 288, 318, 348]
VAL_ANCHOR = 378
FOLDS = [(TRAIN_ANCHORS[:4], 318, 348), (TRAIN_ANCHORS[:5], 348, VAL_ANCHOR)]


def directional_loss(e, d, grp, n_grp, eps: float = 1e-4):
    """Лосс, оптимизирующий то, что реально используется на лидерборде.

    Downstream мы берём ẑ = z₀ + α·d, где α находится аналитически как
    вершина параболы. Значит МАСШТАБ d выбрасывается по построению, и
    остаточная дисперсия равна

        Var(e − α*d) = Var(e) − Cov(e,d)² / Var(d)

    То есть максимизировать надо J(d) = Cov(e,d)²/Var(d), а не минимизировать
    MSE(e, d). Обычный residual-MSE тратит ёмкость на подгонку масштаба
    реализации остатка на обучающих якорях — ровно то, что видно в замере:
    при долгом обучении std(Δz) вырос 0.037 → 0.773, а оптимальный α упал
    1.158 → 0.012, при монотонно падающем лоссе и умирающем переносе.

    Берём знаковую версию −Cov/√Var, а не −Cov²/Var: у неё нормальный
    градиент вблизи нулевой головы, и направление сразу ориентируется
    в правильную сторону, а не в любую из двух.

    Центрирование РАЗДЕЛЬНОЕ ПО ЯКОРЯМ: остаток имеет собственное смещение
    на каждом якоре (замерено +0.033 и −0.038 на 318 и 348), и общее
    центрирование занесло бы межъякорную разницу уровней в ковариацию
    как ложный сигнал.
    """
    import torch
    cnt = torch.zeros(n_grp, device=e.device, dtype=e.dtype).index_add_(
        0, grp, torch.ones_like(e)).clamp(min=1.0)
    me = torch.zeros(n_grp, device=e.device, dtype=e.dtype).index_add_(0, grp, e) / cnt
    md = torch.zeros(n_grp, device=e.device, dtype=e.dtype).index_add_(0, grp, d) / cnt
    ec, dc = e - me[grp], d - md[grp]
    cov = (ec * dc).mean()
    var = (dc * dc).mean()
    return -cov / torch.sqrt(var + eps)


def aux_targets(df, anchor, users, horizon=30):
    """Вспомогательные цели: покупал ли, сколько дней с покупкой, сколько заказов.

    Нужны не ради самих выходов, а чтобы представление описывало механизм
    поведения. Главный таргет — остаток сильной модели — слишком шумный,
    чтобы в одиночку учить энкодер.
    """
    import polars as pl
    t = (df.filter(pl.col("d").is_between(anchor + 1, anchor + horizon))
           .group_by("user_id")
           .agg(n_buy=(pl.col("gmv") > 0).sum(), n_ord=pl.col("to_ord").sum()))
    a = (pl.DataFrame({"user_id": users}).join(t, on="user_id", how="left")
           .with_columns(pl.exclude("user_id").fill_null(0)))
    return (np.log1p(a["n_buy"].to_numpy()).astype("float32"),
            np.log1p(a["n_ord"].to_numpy()).astype("float32"))


class AnchorData:
    """Токены + базовый прогноз + цели одного якоря, выровненные по user_id."""

    def __init__(self, anchor: int, tok_dir: Path, oof_dir: Path, df=None,
                 need_cycles: bool = False):
        import polars as pl
        meta = np.load(tok_dir / f"meta_a{anchor}.npz")
        oof = np.load(oof_dir / f"oof_a{anchor}.npz")
        tu, ou = meta["user_id"], oof["user_id"]
        # пересечение и порядок: молчаливое расхождение здесь дало бы
        # обучение на чужих остатках при формально корректном коде
        common, ti, oi = np.intersect1d(tu, ou, return_indices=True)
        self.anchor, self.user_id = anchor, common
        self.rows = ti
        self.X = np.load(tok_dir / f"x_a{anchor}.npy", mmap_mode="r")
        cf = tok_dir / f"c_a{anchor}.npy"
        if need_cycles and not cf.exists():
            raise FileNotFoundError(
                f"нет токенов циклов {cf}; пересоберите: "
                f"python scripts/build_tokens.py --cycles")
        self.C = np.load(cf, mmap_mode="r") if cf.exists() else None
        self.lengths = meta["lengths"][ti]
        self.z0 = oof["z0"][oi].astype("float32")
        self.dis = (oof["z0_lgb"][oi] - oof["z0_cb"][oi]).astype("float32")
        self.z = np.log1p(oof["y"][oi]).astype("float32")
        self.c = (oof["y"][oi] > 0).astype("float32")
        if df is not None:
            self.n_buy, self.n_ord = aux_targets(df, anchor, pl.Series("user_id", common))
        else:
            self.n_buy = self.n_ord = np.zeros(len(common), dtype="float32")

    def __len__(self):
        return len(self.user_id)


def _load(d, loc, max_len):
    X = np.asarray(d.X[d.rows[loc]], dtype="float32")
    if max_len is not None and X.shape[1] > max_len:
        X = X[:, -max_len:]                  # правый край = ближайшее к якорю
    C = None if d.C is None else np.asarray(d.C[d.rows[loc]], dtype="float32")
    return (d, loc, X, C)


def batches(datasets, bs, rng, feat_idx, shuffle=True, max_len=None,
            homogeneous=False):
    """Батчи по якорям.

    По умолчанию якоря перемешаны: сеть не должна видеть якорь как блок.

    При `homogeneous=True` каждый батч целиком из одного якоря. Это нужно
    направленному лоссу: Cov(e,d)/sqrt(Var(d)) — статистика уровня батча,
    и при смешанном батче на якорь приходится ~450 строк из 2048, отчего
    оценка ковариации шумная. Однородный батч даёт полные 2048 и заодно
    снимает вопрос центрирования: центрирование по батчу становится
    центрированием по якорю по построению.
    """
    if homogeneous:
        chunks = []
        for di, d in enumerate(datasets):
            idx = np.arange(len(d))
            if shuffle:
                rng.shuffle(idx)
            chunks += [(di, np.sort(idx[s:s + bs])) for s in range(0, len(idx), bs)]
        if shuffle:
            rng.shuffle(chunks)
        for di, loc in chunks:
            yield [_load(datasets[di], loc, max_len)], feat_idx
        return

    index = np.array([(di, i) for di, d in enumerate(datasets) for i in range(len(d))],
                     dtype=np.int64)
    if shuffle:
        rng.shuffle(index)
    for s in range(0, len(index), bs):
        chunk = index[s:s + bs]
        yield ([_load(datasets[di], np.sort(chunk[chunk[:, 0] == di, 1]), max_len)
                for di in np.unique(chunk[:, 0])], feat_idx)


def to_torch(parts, feat_idx, dev, max_len):
    import torch
    Xs, gaps, ages, masks, priors, zs, z0s, cs, nbs, nos, cyc = ([] for _ in range(11))
    grp = []
    gi, ai = feat_idx
    for gidx, (d, loc, X, C) in enumerate(parts):
        grp.append(np.full(len(loc), gidx, dtype="int64"))
        if C is not None:
            cyc.append(C)
        L = d.lengths[loc]
        m = np.arange(max_len)[None, :] >= (max_len - np.minimum(L, max_len))[:, None]
        Xs.append(np.delete(X, [gi, ai], axis=2)); gaps.append(X[:, :, gi])
        ages.append(X[:, :, ai]); masks.append(m)
        priors.append(np.stack([d.z0[loc] - d.z0.mean(), d.dis[loc],
                                np.log1p(L) - np.log1p(d.lengths).mean()], 1))
        zs.append(d.z[loc]); z0s.append(d.z0[loc]); cs.append(d.c[loc])
        nbs.append(d.n_buy[loc]); nos.append(d.n_ord[loc])
    T = lambda a, t=torch.float32: torch.as_tensor(np.concatenate(a), dtype=t, device=dev)
    return (T(Xs), T(gaps), T(ages), T(masks, torch.bool), T(priors),
            T(zs), T(z0s), T(cs), T(nbs), T(nos),
            T(cyc) if cyc else None, T(grp, torch.int64), len(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, default=Path("artifacts/neural/tokens"))
    ap.add_argument("--oof", type=Path, default=Path("artifacts/neural"))
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lambda-delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=-1, help="-1 = оба")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="чекпоинт этапа A; ищется как <путь>_fold{i}.pt")
    ap.add_argument("--freeze-epochs", type=int, default=2,
                    help="эпох с замороженным энкодером после претрейна")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--loss", choices=("mse", "dir"), default="mse",
                    help="mse — residual-MSE; dir — направленный Cov/sqrt(Var)")
    ap.add_argument("--accum", type=int, default=4,
                    help="однородных батчей на один шаг оптимизатора")
    ap.add_argument("--lambda-mse", type=float, default=0.1,
                    help="вес слабого residual-MSE при --loss dir")
    ap.add_argument("--production", action="store_true",
                    help="обучать на ВСЕХ якорях включая 348 и 378, без оценки; "
                         "число эпох и alpha берутся из проверенного фолдового прогона")
    ap.add_argument("--tb", type=Path, default=None,
                    help="каталог логов TensorBoard")
    ap.add_argument("--eval-every", type=int, default=2,
                    help="каждые N эпох мерить выигрыш на якоре подбора α")
    ap.add_argument("--cycles", action="store_true",
                    help="включить вторую шкалу времени: токены покупочных циклов")
    ap.add_argument("--out", type=Path, default=Path("artifacts/neural/gapgru.json"))
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    import torch
    from torch import nn
    from ecup import load_panel
    from ecup.gapgru import GapGRUConfig, make_model, pick_device
    from ecup.tokens import TOKEN_FEATURES

    gi, ai = TOKEN_FEATURES.index("gap"), TOKEN_FEATURES.index("age")
    dev = pick_device()
    df = load_panel()
    print(f"устройство {dev} · длина {a.max_len}", flush=True)

    def make_writer(tag):
        if a.tb is None:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("  tensorboard не установлен, логи пропускаются", flush=True)
            return None
        return SummaryWriter(str(a.tb / tag))

    cache = {}
    def get(anchor):
        if anchor not in cache:
            cache[anchor] = AnchorData(anchor, a.tokens, a.oof, df,
                                       need_cycles=a.cycles)
            print(f"  якорь {anchor}: {len(cache[anchor]):,} пользователей", flush=True)
        return cache[anchor]

    results = []
    # Боевой режим: якоря 348 и 378 в фолдовой схеме тратятся на подбор α
    # и на оценку, то есть два самых свежих окна не участвуют в обучении.
    # Для прогноза на 408 это чистая потеря. Число эпох и α сюда переносятся
    # из фолдового прогона — подбирать их здесь не на чем, и это осознанное
    # ограничение режима, а не упущение.
    folds = ([(TRAIN_ANCHORS + [VAL_ANCHOR], None, None)] if a.production
             else FOLDS if a.fold < 0 else [FOLDS[a.fold]])
    for fi, (tr_anchors, alpha_anchor, test_anchor) in enumerate(folds):
        if a.production:
            print(f"\n=== боевой режим: обучение {tr_anchors}, оценки нет ===",
                  flush=True)
        else:
            print(f"\n=== фолд {fi}: обучение {tr_anchors} · α на {alpha_anchor} · "
                  f"оценка на {test_anchor} ===", flush=True)
        tr = [get(x) for x in tr_anchors]
        cfg = GapGRUConfig(n_features=len(TOKEN_FEATURES) - 2, max_len=a.max_len,
                           lambda_delta=a.lambda_delta, lr=a.lr,
                           batch_size=a.batch_size, epochs=a.epochs, seed=a.seed,
                           use_cycles=a.cycles)
        model = make_model(cfg).to(dev)
        if a.init_from is not None:
            src = Path(f"{a.init_from}_fold{fi}.pt")
            if not src.exists():
                raise FileNotFoundError(f"нет чекпоинта этапа A: {src}")
            sd = torch.load(src, map_location=dev)["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  претрейн загружен из {src.name}"
                  + (f" (не найдено: {len(missing)})" if missing else ""), flush=True)
        # Голова поправки должна стартовать с нуля даже после претрейна:
        # иначе первый же шаг сдвинет прогноз, не объяснив ошибку.
        nn.init.zeros_(model.head_dz.weight); nn.init.zeros_(model.head_dz.bias)
        enc = [p_ for n_, p_ in model.named_parameters() if not n_.startswith(("trunk", "head_dz"))]
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        n_steps = cfg.epochs * (sum(len(d) for d in tr) // cfg.batch_size + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                                    total_steps=n_steps)
        bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
        rng = np.random.default_rng(cfg.seed)
        tb = make_writer(f"stageB_fold{fi}")

        def predict(d):
            model.eval(); out = []
            with torch.no_grad():
                for parts, fx in batches([d], 4096, rng, (gi, ai), shuffle=False,
                                         max_len=a.max_len):
                    X, g, ag, m, pr, _z, _z0, _c, _nb, _no, cy, _gr, _ng = to_torch(
                        parts, fx, dev, a.max_len)
                    out.append(model(X, g, ag, m, pr,
                                     cycles=cy if a.cycles else None)[0]
                               .float().cpu().numpy())
            model.train()
            return np.concatenate(out)

        # Кривая сходимости по якорю подбора α. Он уже используется для
        # подбора одного скаляра, поэтому выбор эпохи по нему согласован
        # и не трогает тестовый якорь. Без этой кривой число эпох остаётся
        # прикидкой — ровно той ошибкой, что была с числом деревьев.
        da_early = None if a.production else get(alpha_anchor)
        best = {"gain": -1.0, "epoch": 0, "state": None}
        curve = []
        # Устойчивая часть направления повторяется между соседними
        # чекпоинтами, идиосинкратический шум — нет. Поэтому копим
        # НОРМИРОВАННЫЕ направления и усредняем: масштаб каждого произволен,
        # его всё равно поглотит α. Один подобранный скаляр на всё,
        # в отличие от стекинга с весом на каждого участника.
        acc_a, acc_t, n_acc = None, None, 0
        # В боевом режиме тестового якоря нет по построению — там нечего
        # оценивать, и попытка загрузить его токены даёт meta_aNone.npz.
        dt_test = None if a.production else get(test_anchor)

        for ep in range(cfg.epochs):
            # Энкодер после претрейна сначала заморожен: остаток очень шумный,
            # и разморозка с первого шага стёрла бы выученное поведение
            # градиентами по шуму.
            frozen = a.init_from is not None and ep < a.freeze_epochs
            for p_ in enc:
                p_.requires_grad_(not frozen)
            t0, tot, nb = time.perf_counter(), 0.0, 0
            step = 0
            for parts, fx in batches(tr, cfg.batch_size, rng, (gi, ai),
                                     max_len=a.max_len,
                                     homogeneous=(a.loss == "dir")):
                X, g, ag, m, pr, z, z0, c, nbuy, nord, cy, grp, ngrp = to_torch(
                    parts, fx, dev, a.max_len)
                if a.loss != "dir":
                    opt.zero_grad(set_to_none=True)
                dz, aux = model(X, g, ag, m, pr, cycles=cy if a.cycles else None)
                if a.loss == "dir":
                    # штраф на масштаб здесь не нужен и вреден: целевая
                    # функция инвариантна к масштабу по построению
                    loss = (directional_loss(z - z0, dz, grp, ngrp)
                            + a.lambda_mse * mse(z0 + dz, z))
                else:
                    loss = mse(z0 + dz, z) + cfg.lambda_delta * (dz ** 2).mean()
                loss = loss + cfg.aux_weights["p"] * bce(aux["p"], c)
                loss = loss + cfg.aux_weights["n_buy"] * mse(aux["n_buy"], nbuy)
                loss = loss + cfg.aux_weights["n_ord"] * mse(aux["n_ord"], nord)
                # Однородные батчи делают каждый шаг видящим один якорь.
                # Чтобы градиент не гулял вслед за идиосинкразией окна,
                # накапливаем несколько якорей на один шаг оптимизатора.
                acc = a.accum if a.loss == "dir" else 1
                (loss / acc).backward()
                step += 1
                if step % acc == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step()
                    opt.zero_grad(set_to_none=True)
                tot += float(loss.item()); nb += 1
            if tb is not None:
                tb.add_scalar("loss/train", tot / max(nb, 1), ep + 1)
                tb.add_scalar("lr", sched.get_last_lr()[0], ep + 1)
                tb.add_scalar("frozen", int(frozen), ep + 1)
            print(f"  эпоха {ep+1}/{cfg.epochs}  loss {tot/max(nb,1):.5f}"
                  f"{'  [энкодер заморожен]' if frozen else ''}  "
                  f"{time.perf_counter()-t0:.0f}с", flush=True)
            if da_early is not None and ((ep + 1) % a.eval_every == 0
                                          or ep == cfg.epochs - 1):
                dz_e = predict(da_early)
                e_e = da_early.z - da_early.z0
                De = float((dz_e ** 2).mean())
                al = float((e_e * dz_e).mean()) / max(De, 1e-12)
                g = float(e_e.std() - (e_e - al * dz_e).std())
                curve.append((ep + 1, g, al, float(dz_e.std())))
                if tb is not None:
                    # главная кривая: не лосс, а переносимый выигрыш —
                    # именно по ней принимается решение об эпохе
                    tb.add_scalar("holdout/gain", g, ep + 1)
                    tb.add_scalar("holdout/alpha", al, ep + 1)
                    tb.add_scalar("holdout/std_dz", float(dz_e.std()), ep + 1)
                    tb.add_histogram("holdout/dz", dz_e, ep + 1)
                print(f"    якорь {alpha_anchor}: выигрыш {g:+.5f} · α {al:+.4f} · "
                      f"std(Δz) {dz_e.std():.4f}", flush=True)
                nz = (dz_e - dz_e.mean()) / max(dz_e.std(), 1e-9)
                acc_a = nz if acc_a is None else acc_a + nz
                nzt = predict(dt_test)
                nzt = (nzt - nzt.mean()) / max(nzt.std(), 1e-9)
                acc_t = nzt if acc_t is None else acc_t + nzt
                n_acc += 1
                if g > best["gain"]:
                    best = {"gain": g, "epoch": ep + 1,
                            "state": {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}}
            if a.ckpt is not None:
                a.ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "fold": fi, "epoch": ep + 1},
                           Path(f"{a.ckpt}_fold{fi}.pt"))

        if a.production:
            Path(f"{a.ckpt}_prod.pt").parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "fold": "prod",
                        "epoch": cfg.epochs, "anchors": tr_anchors},
                       Path(f"{a.ckpt}_prod.pt"))
            print(f"  сохранено {a.ckpt}_prod.pt (эпоха {cfg.epochs})", flush=True)
            results.append(dict(mode="production", epochs=cfg.epochs,
                                anchors=tr_anchors, seed=cfg.seed))
            del model; gc.collect()
            continue

        # Берём эпоху с лучшим выигрышем на якоре подбора, а не последнюю:
        # кривая может пройти максимум и начать переобучаться.
        if best["state"] is not None and best["epoch"] != cfg.epochs:
            model.load_state_dict(best["state"])
            print(f"  восстановлена эпоха {best['epoch']} "
                  f"(выигрыш {best['gain']:+.5f} против {curve[-1][1]:+.5f} на последней)",
                  flush=True)
        if a.ckpt is not None:
            torch.save({"model": model.state_dict(), "fold": fi,
                        "epoch": best["epoch"], "curve": curve},
                       Path(f"{a.ckpt}_fold{fi}.pt"))

        # α на отдельном якоре: парабола MSE(α) = MSE(0) - 2αC + α²D
        da = get(alpha_anchor); dz_a = predict(da)
        e = da.z - da.z0
        D = float((dz_a ** 2).mean()); C = float((e * dz_a).mean())
        alpha = C / max(D, 1e-12)
        dt = dt_test; dz_t = predict(dt)
        base = float((dt.z - dt.z0).std())
        got = float((dt.z - dt.z0 - alpha * dz_t).std())

        # То же самое, но по усреднённому направлению чекпоинтов
        ens_gain = None
        if n_acc > 1:
            da_e, dt_e = acc_a / n_acc, acc_t / n_acc
            De = float((da_e ** 2).mean())
            al_e = float((e * da_e).mean()) / max(De, 1e-12)
            ens_gain = base - float((dt.z - dt.z0 - al_e * dt_e).std())
            print(f"  усреднение {n_acc} чекпоинтов: α {al_e:+.4f} · "
                  f"выигрыш {ens_gain:+.5f} против {base - got:+.5f} "
                  f"у одиночного", flush=True)
        row = dict(fold=fi, alpha=alpha, D=D, C=C, shape_base=base,
                   best_epoch=best["epoch"], curve=curve,
                   gain_ckpt_ens=ens_gain, n_ckpt=n_acc,
                   shape_corrected=got, gain=base - got,
                   alpha_anchor_gain=float((e).std() - (e - alpha * dz_a).std()),
                   std_dz=float(dz_t.std()))
        # Разрыв между выигрышем на якоре подбора и на тестовом — прямая мера
        # переносимости. Большой выигрыш на первом при нуле на втором означает,
        # что α описал особенность одного окна, а не общий сигнал.
        print(f"  α = {alpha:+.4f} · std(Δz) = {row['std_dz']:.4f}\n"
              f"  на {alpha_anchor} (подбор α): выигрыш {row['alpha_anchor_gain']:+.5f}\n"
              f"  на {test_anchor} (оценка):   {base:.5f} → {got:.5f}  "
              f"выигрыш {row['gain']:+.5f}\n"
              f"  перенос: {100 * row['gain'] / max(row['alpha_anchor_gain'], 1e-9):.0f}% "
              f"выигрыша якоря подбора", flush=True)
        if tb is not None:
            tb.add_scalar("final/gain_test", row["gain"], 0)
            tb.add_scalar("final/alpha", alpha, 0)
            tb.close()
        results.append(row)
        del model; gc.collect()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== итог ===")
    for r in results:
        if "gain" in r:
            print(f"  фолд {r['fold']}: выигрыш {r['gain']:+.5f} (α={r['alpha']:+.4f})")
    if any("gain" not in r for r in results):
        print("  боевой режим: оценки нет по построению")
        print("ГОТОВО"); return
    ok = all(r["gain"] > 0.0002 for r in results)
    print(f"  критерий (плюс на ОБОИХ фолдах, > 0.0002): "
          f"{'ПРОЙДЕН' if ok else 'не пройден'}")
    print("ГОТОВО")


if __name__ == "__main__":
    main()
