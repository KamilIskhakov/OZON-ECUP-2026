"""Сборка токенов на диск: по одному файлу на якорь, fp16, memmap-совместимо.

Держать всё в памяти нельзя: 7 якорей × 233k пользователей × 192 токена ×
20 признаков в fp16 — это 12.5 ГБ. Поэтому каждый якорь пишется отдельно
и читается стримингом. На этом уже три раза ловился тихий OOM, когда
таблицы признаков держались для всех комбинаций сразу.

Запуск:
    python scripts/build_tokens.py --max-len 192 --out artifacts/neural/tokens
    python scripts/build_tokens.py --max-len 96 --users 80000   # локальный пилот
"""
from __future__ import annotations

import argparse, gc, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--max-history", type=int, default=300)
    ap.add_argument("--users", type=int, default=0, help="0 = все отобранные")
    ap.add_argument("--out", type=Path, default=Path("artifacts/neural/tokens"))
    ap.add_argument("--cycles", action="store_true", help="дополнительно циклы покупок")
    a = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    from ecup import SplitConfig, load_panel, selected_users
    from ecup.tokens import build_tokens, cycle_tokens, N_TOKEN_FEATURES

    a.out.mkdir(parents=True, exist_ok=True)
    sp = SplitConfig(max_history=a.max_history, n_train_anchors=6, with_state=True)
    anchors = sp.train_anchors() + [sp.val_anchor, sp.final_anchor]
    df = load_panel()
    print(f"якоря: {anchors}\nдлина {a.max_len} · признаков {N_TOKEN_FEATURES}", flush=True)

    for anchor in anchors:
        t = time.perf_counter()
        u = selected_users(df, anchor)
        if a.users:
            # подвыборка детерминированная: один и тот же пользователь должен
            # попадать в выборку на всех якорях, иначе остаточный таргет
            # и признаки перестанут соответствовать друг другу
            arr = u.to_numpy()
            keep = arr[(arr.astype(np.uint64) * 2654435761 % 1000000)
                       < a.users / len(arr) * 1000000]
            import polars as pl
            u = pl.Series("user_id", keep)
        X, L = build_tokens(df, anchor, u, a.max_history, a.max_len)
        np.save(a.out / f"x_a{anchor}.npy", X)
        np.savez(a.out / f"meta_a{anchor}.npz", user_id=u.to_numpy(), lengths=L)
        msg = (f"якорь {anchor}: {X.shape} → {X.nbytes / 2**30:.2f} ГБ · "
               f"усечено {100 * (L >= a.max_len).mean():.1f}% · {time.perf_counter()-t:.0f}с")
        if a.cycles:
            C, LC = cycle_tokens(df, anchor, u, a.max_history)
            np.save(a.out / f"c_a{anchor}.npy", C.astype("float16"))
            msg += f" · циклы {C.shape}"
        print(msg, flush=True)
        del X, L; gc.collect()
    print("ГОТОВО")


if __name__ == "__main__":
    main()
