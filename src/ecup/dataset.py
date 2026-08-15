"""Сборка обучающего набора из нескольких якорей."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import SplitConfig, Windows, anchor_label
from .data import anchor_population, make_target
from .features import build_features, feature_names
from .state import state_features


@dataclass
class AnchorLevel:
    """Уровень окна, разложенный на две маржи.

    Так как log1p(0) = 0, безусловное среднее раскладывается точно:
    ℓ = p̄ · ℓ⁺. Экстенсивная маржа p̄ гуляет между сезонами на порядок
    сильнее интенсивной ℓ⁺, поэтому нормировать головы надо раздельно.
    """

    p_bar: float             # доля покупателей
    l_plus: float            # mean log1p(y) среди покупавших
    l: float                 # безусловное среднее = p_bar * l_plus


@dataclass
class AnchorFrame:
    """Признаки и таргет одного якоря."""

    anchor: int
    X: pl.DataFrame          # включает user_id
    y: np.ndarray
    level: AnchorLevel

    def __len__(self) -> int:
        return self.X.height


def build_anchor(
    df: pl.DataFrame,
    anchor: int,
    split: SplitConfig,
    windows: Windows | None = None,
    with_target: bool = True,
) -> AnchorFrame:
    users = anchor_population(df, anchor, split.apply_selection)
    X = build_features(df, anchor, users, max_history=split.max_history, windows=windows)
    if split.with_state:
        X = X.join(state_features(df, anchor, users, span=split.max_history),
                   on="user_id", how="left")
    nan_lvl = AnchorLevel(float("nan"), float("nan"), float("nan"))
    if not with_target:
        return AnchorFrame(anchor=anchor, X=X, y=np.zeros(X.height), level=nan_lvl)
    tgt = make_target(df, anchor, users, horizon=split.horizon)
    y = tgt["y"].to_numpy()
    z = np.log1p(y)
    pos = y > 0
    lvl = AnchorLevel(p_bar=float(pos.mean()),
                      l_plus=float(z[pos].mean()) if pos.any() else 0.0,
                      l=float(z.mean()))
    return AnchorFrame(anchor=anchor, X=X, y=y, level=lvl)


def build_training_set(
    df: pl.DataFrame,
    anchors: list[int],
    split: SplitConfig,
    windows: Windows | None = None,
    verbose: bool = True,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[int, "AnchorLevel"]]:
    """Склеить якоря в один обучающий набор.

    Окна признаков соседних якорей перекрываются, а таргет-окна — нет,
    поэтому утечки ответа между примерами не возникает.

    Возвращает и уровни якорей: они нужны как офсеты при обучении и как
    ручки при предсказании.
    """
    frames, ys, anchor_ids, levels = [], [], [], {}
    for a in anchors:
        t0 = time.perf_counter()
        af = build_anchor(df, a, split, windows)
        levels[a] = af.level
        frames.append(af.X)
        ys.append(af.y)
        anchor_ids.append(np.full(len(af), a, dtype=np.int32))
        if verbose:
            print(f"  якорь {anchor_label(a)}: {len(af):>7,} строк · "
                  f"p̄ {af.level.p_bar:.4f} · ℓ⁺ {af.level.l_plus:.4f} · "
                  f"ℓ {af.level.l:.4f} · {time.perf_counter()-t0:.1f}с")

    X = pl.concat(frames, how="vertical")
    return X, np.concatenate(ys), np.concatenate(anchor_ids), levels


def anchor_offsets(anchor_ids: np.ndarray,
                   levels: dict[int, "AnchorLevel"]) -> tuple[np.ndarray, np.ndarray]:
    """(init_score для классификатора, офсет уровня для регрессии) по строкам."""
    pb = np.array([levels[a].p_bar for a in anchor_ids])
    lp = np.array([levels[a].l_plus for a in anchor_ids])
    return np.log(pb / (1.0 - pb)), lp


def to_matrix(X: pl.DataFrame, feats: list[str] | None = None,
              drop_anchor_calendar: bool = True) -> tuple[np.ndarray, list[str]]:
    feats = feats or feature_names(X, drop_anchor_calendar=drop_anchor_calendar)
    return X.select(feats).to_numpy().astype("float32"), feats


def anchor_weights(anchor_ids: np.ndarray, half_life: float = 90.0) -> np.ndarray:
    """Веса примеров: свежие якоря важнее, вес падает вдвое каждые `half_life` дней.

    Мотивация — дрейф между якорями (01_eda §5): чем дальше якорь, тем менее
    он похож на боевой по составу и по уровню спроса.
    """
    newest = anchor_ids.max()
    age = (newest - anchor_ids).astype("float64")
    return np.power(0.5, age / half_life)
