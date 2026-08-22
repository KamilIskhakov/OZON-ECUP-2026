"""Многомасштабные патчи: недельные и месячные агрегаты вместо событий.

Gap-GRU видит последовательность активных дней и паузы между ними. Это
эффективно для событийной динамики, но форму активности на годовой шкале
он видит лишь косвенно — годовой фильтр вытащил её вручную заданной
гармоникой, и это дало больше, чем вся нейросетевая ветка после первого
Gap-GRU.

Здесь два потока: 26 недельных патчей по свежей истории и 12 месячных
по всему году. Патч — это агрегат, поэтому дневной тензор материализовать
не нужно: 250k × 38 × 8 против 250k × 365 × 8.

Каждый патч подаётся ДВАЖДЫ: как есть и с вычтенным средним по потоку.
Это не украшение — измерено, что уровень маскирует форму: ковариация
годового сигнала с остатком выросла втрое после удаления уровня.
"""
from __future__ import annotations

import numpy as np
import polars as pl

CH = ("gmv", "orders", "carts", "searches", "gmv_search", "gmv_cat", "buydays", "actdays")
N_CH = len(CH)
N_WEEK, WEEK = 26, 7          # свежий поток: 182 дня
N_MON, MON = 12, 30           # длинный поток: 360 дней


def _agg(df: pl.DataFrame, uid: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """(N, len(spans), N_CH) — агрегаты по заданным окнам."""
    u = pl.Series("user_id", uid)
    out = np.zeros((len(uid), len(spans), N_CH), dtype="float32")
    for k, (lo, hi) in enumerate(spans):
        if hi < 0:
            continue
        t = (df.filter(pl.col("user_id").is_in(u) & pl.col("d").is_between(max(lo, 0), hi))
               .group_by("user_id")
               .agg(gmv=pl.col("gmv").sum(), orders=pl.col("to_ord").sum(),
                    carts=pl.col("to_cart").sum(), searches=pl.col("searches").sum(),
                    gmv_search=pl.col("gmv_search").sum(), gmv_cat=pl.col("gmv_cat").sum(),
                    buydays=(pl.col("gmv") > 0).sum(), actdays=pl.len()))
        r = (pl.DataFrame({"user_id": uid}).join(t, on="user_id", how="left")
               .with_columns(pl.exclude("user_id").fill_null(0)).sort("user_id"))
        for c, name in enumerate(CH):
            out[:, k, c] = np.log1p(r[name].to_numpy().astype("float64"))
    return out


def build_patches(df: pl.DataFrame, anchor: int, uid: np.ndarray) -> dict:
    """Два потока плюс календарная позиция каждого патча относительно цели.

    Фаза считается от центра ЦЕЛЕВОГО окна, а не от якоря: именно так была
    зафиксирована фаза годового фильтра, и именно она физически осмысленна.
    """
    wk = [(anchor - WEEK * (k + 1) + 1, anchor - WEEK * k) for k in range(N_WEEK)][::-1]
    mn = [(anchor - MON * (k + 1) + 1, anchor - MON * k) for k in range(N_MON)][::-1]
    Xw, Xm = _agg(df, uid, wk), _agg(df, uid, mn)

    def pos(spans):
        c = np.array([(lo + hi) / 2 for lo, hi in spans], dtype="float32")
        dt = c - (anchor + 15.5)                      # расстояние до центра цели
        return np.stack([np.sin(2*np.pi*dt/365), np.cos(2*np.pi*dt/365),
                         np.sin(2*np.pi*dt/7), np.cos(2*np.pi*dt/7),
                         np.log1p(-dt) / 6.0], 1).astype("float32")
    return {"week": Xw, "mon": Xm, "pos_week": pos(wk), "pos_mon": pos(mn)}


def add_shape(X: np.ndarray) -> np.ndarray:
    """(N, T, C) -> (N, T, 2C): исходный канал и центрированный по потоку."""
    return np.concatenate([X, X - X.mean(1, keepdims=True)], axis=2)
