"""Событийная токенизация истории: один токен = один день с активностью.

Зачем не 409 дневных токенов. Медиана активных дней в окне 300 дней — 83,
то есть три четверти позиций дневного представления заняты нулями. Сеть
тратила бы ёмкость на то, чтобы выучить «нулевой день ничего не значит»,
и при этом отсутствие пользователя на 17 дней выглядело бы как семнадцать
одинаковых нулей вместо одного числа.

Здесь пауза кодируется явно: у каждого токена есть `gap` — сколько дней
прошло с предыдущего события. Это ровно то, что в GRU-D подаётся в динамику
рекуррентности: сам паттерн отсутствия наблюдений информативен. У нас он
информативен даже сильнее, потому что «нет строки» — не пропуск измерения,
а факт поведения.

Выравнивание по ПРАВОМУ краю: последний токен всегда ближайшее к якорю
событие. При усечении теряется старая история, а не свежая.
"""
from __future__ import annotations

import numpy as np
import polars as pl

# Числовые каналы: подаются как log1p. Разделение search/cat сквозное —
# задача называется Search LTV, и вклад каналов различать обязательно.
TOKEN_NUM: tuple[str, ...] = (
    "gmv", "gmv_search", "gmv_cat", "to_cart", "to_ord", "searches",
    "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
)
# Бинарные признаки дня: наличие канала и наличие перехода в корзину/заказ.
TOKEN_FLAG: tuple[str, ...] = (
    "search", "cat", "has_search_to_cart", "has_search_to_ord",
    "has_cat_to_cart", "has_cat_to_ord",
)
# Временные: пауза до события, возраст события, фаза недели.
TOKEN_TIME: tuple[str, ...] = ("gap", "age", "dow_sin", "dow_cos")

TOKEN_FEATURES: tuple[str, ...] = TOKEN_NUM + TOKEN_FLAG + TOKEN_TIME
N_TOKEN_FEATURES = len(TOKEN_FEATURES)
MAX_LEN = 192


def build_tokens(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    max_history: int = 300,
    max_len: int = MAX_LEN,
    dtype: str = "float16",
) -> tuple[np.ndarray, np.ndarray]:
    """(X, lengths): X — (N, max_len, F) с паддингом слева, lengths — (N,).

    Паддинг нулями слева, поэтому маска восстанавливается из `lengths`
    и хранить её отдельно не нужно.
    """
    lo = anchor - max_history + 1
    h = (
        df.lazy()
          .filter(pl.col("user_id").is_in(users) & pl.col("d").is_between(lo, anchor))
          .sort(["user_id", "d"])
          .with_columns(
              # пауза до предыдущего события; у самого старого токена отсчёт
              # ведётся от начала окна, иначе первая пауза была бы неопределена
              gap=(pl.col("d") - pl.col("d").shift(1).over("user_id"))
                    .fill_null(pl.col("d") - lo + 1).cast(pl.Int32),
              age=(anchor - pl.col("d")).cast(pl.Int32),
              rk=pl.col("d").rank("ordinal", descending=True).over("user_id"),
          )
          .filter(pl.col("rk") <= max_len)
          .with_columns(pos=(max_len - pl.col("rk")).cast(pl.Int32))
          .collect()
    )

    pos_of = {u: i for i, u in enumerate(users.to_numpy())}
    ri = np.fromiter((pos_of[u] for u in h["user_id"].to_numpy()),
                     dtype=np.int64, count=h.height)
    ci = h["pos"].to_numpy().astype(np.int64)

    X = np.zeros((len(users), max_len, N_TOKEN_FEATURES), dtype=dtype)
    k = 0
    for c in TOKEN_NUM:
        X[ri, ci, k] = np.log1p(h[c].to_numpy().astype("float32")); k += 1
    for c in TOKEN_FLAG:
        X[ri, ci, k] = h[c].to_numpy().astype("float32"); k += 1
    X[ri, ci, k] = np.log1p(h["gap"].to_numpy().astype("float32")); k += 1
    X[ri, ci, k] = np.log1p(h["age"].to_numpy().astype("float32")); k += 1
    dow = h["dow"].to_numpy().astype("float32") * (2 * np.pi / 7.0)
    X[ri, ci, k] = np.sin(dow); k += 1
    X[ri, ci, k] = np.cos(dow); k += 1
    assert k == N_TOKEN_FEATURES

    lengths = np.zeros(len(users), dtype=np.int32)
    cnt = h.group_by("user_id").agg(n=pl.len())
    cnt = (pl.DataFrame({"user_id": users}).join(cnt, on="user_id", how="left")
             .with_columns(pl.col("n").fill_null(0)))
    lengths[:] = cnt["n"].to_numpy()
    return X, lengths


def cycle_tokens(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    max_history: int = 300,
    max_cycles: int = 16,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    """Токены покупочных циклов: отрезок между соседними покупками.

    Признаки ритма покупок дали одну из лучших прибавок среди семей
    (+0.0007), но сжимают всю траекторию в медиану и разброс интервалов.
    Последовательности 7,8,7,8,9,27 и 11,10,11,10,11,10 имеют близкую
    медиану и совершенно разную динамику; здесь она сохраняется.

    Признаки цикла: длина, накопленные внутри него активность, поиски,
    корзины, заказы и GMV покупки, которой цикл закрылся.
    """
    lo = anchor - max_history + 1
    h = (df.lazy()
           .filter(pl.col("user_id").is_in(users) & pl.col("d").is_between(lo, anchor))
           .sort(["user_id", "d"])
           .with_columns(buy=(pl.col("gmv") > 0).cast(pl.Int8))
           # номер цикла = сколько покупок уже случилось строго раньше
           .with_columns(cyc=pl.col("buy").cum_sum().over("user_id") - pl.col("buy"))
           .group_by(["user_id", "cyc"])
           .agg(d_lo=pl.col("d").min(), d_hi=pl.col("d").max(),
                n_act=pl.len(), searches=pl.col("searches").sum(),
                to_cart=pl.col("to_cart").sum(), to_ord=pl.col("to_ord").sum(),
                gmv=pl.col("gmv").sum(), closed=pl.col("buy").max())
           .sort(["user_id", "cyc"])
           .with_columns(rk=pl.col("cyc").rank("ordinal", descending=True)
                                .over("user_id"))
           .filter(pl.col("rk") <= max_cycles)
           .with_columns(pos=(max_cycles - pl.col("rk")).cast(pl.Int32))
           .collect())

    cols = ("n_act", "searches", "to_cart", "to_ord", "gmv")
    F = len(cols) + 3                       # + длина цикла, разрыв, признак закрытия
    pos_of = {u: i for i, u in enumerate(users.to_numpy())}
    ri = np.fromiter((pos_of[u] for u in h["user_id"].to_numpy()),
                     dtype=np.int64, count=h.height)
    ci = h["pos"].to_numpy().astype(np.int64)
    X = np.zeros((len(users), max_cycles, F), dtype=dtype)
    k = 0
    for c in cols:
        X[ri, ci, k] = np.log1p(h[c].to_numpy().astype("float32")); k += 1
    X[ri, ci, k] = np.log1p((h["d_hi"] - h["d_lo"] + 1).to_numpy().astype("float32")); k += 1
    X[ri, ci, k] = np.log1p((anchor - h["d_hi"]).to_numpy().astype("float32")); k += 1
    X[ri, ci, k] = h["closed"].to_numpy().astype("float32"); k += 1
    lengths = np.zeros(len(users), dtype=np.int32)
    cnt = (pl.DataFrame({"user_id": users})
             .join(h.group_by("user_id").agg(n=pl.len()), on="user_id", how="left")
             .with_columns(pl.col("n").fill_null(0)))
    lengths[:] = cnt["n"].to_numpy()
    return X, lengths
