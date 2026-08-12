"""Загрузка панели активности и производные от неё выборки пользователей."""
from __future__ import annotations

import gc

import polars as pl

from .config import (COMPACT_PARQUET, DATA_WORK, DAY0, HORIZON, N_DAYS,
                     SELECTION_BLOCK_LEN, SELECTION_BLOCKS, SELECTION_SPAN,
                     TRAIN_PARQUET)

# Даункаст: исходник в Int64/Float64 занимает ~4.1 GB, после приведения ~1.1 GB.
# Максимальный дневной GMV — 73 830, Float32 держит 7 значащих цифр, потери нет.
SCHEMA_CAST: dict[str, pl.DataType] = {
    "user_id": pl.UInt32,
    "search": pl.Int8, "cat": pl.Int8,
    "has_search_to_cart": pl.Int8, "has_search_to_ord": pl.Int8,
    "has_cat_to_cart": pl.Int8, "has_cat_to_ord": pl.Int8,
    "search_to_cart": pl.UInt16, "search_to_ord": pl.UInt16,
    "cat_to_cart": pl.UInt16, "cat_to_ord": pl.UInt16,
    "to_cart": pl.UInt16, "to_ord": pl.UInt16, "searches": pl.UInt16,
    "gmv_search": pl.Float32, "gmv_cat": pl.Float32, "gmv": pl.Float32,
}


def load_panel(rebuild: bool = False) -> pl.DataFrame:
    """Дневная панель активности с целочисленным индексом дня `d`.

    Дата заменена на `d = (event_date - 2025-01-01).days`: все оконные условия
    становятся сравнением Int16 вместо арифметики дат.
    """
    DATA_WORK.mkdir(exist_ok=True)
    if COMPACT_PARQUET.exists() and not rebuild:
        return pl.read_parquet(COMPACT_PARQUET)

    raw = pl.read_parquet(TRAIN_PARQUET)
    df = (
        raw.with_columns([pl.col(c).cast(t) for c, t in SCHEMA_CAST.items()])
           .with_columns(
               d=((pl.col("event_date") - pl.lit(DAY0)).dt.total_days()).cast(pl.Int16),
               dow=(pl.col("event_date").dt.weekday() - 1).cast(pl.Int8),  # 0 = понедельник
           )
           .drop("event_date")
           .sort(["user_id", "d"])
    )
    del raw
    gc.collect()
    df.write_parquet(COMPACT_PARQUET, compression="zstd")
    return df


def all_users(df: pl.DataFrame) -> pl.Series:
    return df["user_id"].unique().sort()


def selected_users(
    df: pl.DataFrame,
    anchor: int,
    blocks: int = SELECTION_BLOCKS,
    block_len: int = SELECTION_BLOCK_LEN,
) -> pl.Series:
    """Пользователи, проходящие правило отбора организаторов на момент `anchor`.

    Правило восстановлено в 01_eda.ipynb §3.1: нужно хотя бы одно событие
    в каждом из `blocks` подряд идущих окон длины `block_len`, отсчитанных
    назад от якоря. На боевом якоре проходят ровно все 250 000 пользователей.

    Зачем это на обучающих якорях: без фильтра в трейн попадает популяция,
    которой в тесте не существует (давно не заходившие, у кого таргет почти
    гарантированно нулевой), и классификатор занижает P(y>0).
    """
    lo = anchor - blocks * block_len + 1
    s = (
        df.filter(pl.col("d").is_between(lo, anchor))
          .with_columns(b=((pl.col("d") - lo) // block_len).cast(pl.Int8))
          .group_by("user_id")
          .agg(nb=pl.col("b").n_unique())
    )
    return s.filter(pl.col("nb") == blocks)["user_id"].sort()


def make_target(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    horizon: int = HORIZON,
) -> pl.DataFrame:
    """Суммарный GMV за (anchor, anchor + horizon] для каждого пользователя.

    Отсутствие строк в окне означает нулевой таргет, а не пропуск, — поэтому
    left join с заполнением нулями, а не inner.
    """
    if anchor + horizon > N_DAYS - 1:
        raise ValueError(
            f"таргет-окно якоря {anchor} выходит за пределы данных "
            f"(нужен день {anchor + horizon}, последний доступный {N_DAYS - 1})"
        )
    t = (
        df.filter(pl.col("d").is_between(anchor + 1, anchor + horizon))
          .group_by("user_id")
          .agg(y=pl.col("gmv").sum().cast(pl.Float64))
    )
    return (
        pl.DataFrame({"user_id": users})
          .join(t, on="user_id", how="left")
          .with_columns(pl.col("y").fill_null(0.0))
          .sort("user_id")
    )


def anchor_population(
    df: pl.DataFrame,
    anchor: int,
    apply_selection: bool = True,
) -> pl.Series:
    """Популяция якоря: с фильтром отбора или все пользователи."""
    return selected_users(df, anchor) if apply_selection else all_users(df)


def earliest_valid_anchor(max_history: int) -> int:
    """Ранняя граница якоря: нужна и глубина признаков, и окно правила отбора."""
    return max(max_history, SELECTION_SPAN) - 1
