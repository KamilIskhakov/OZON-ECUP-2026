"""Метрика соревнования и правило склейки hurdle.

RMSLE — это RMSE в пространстве z = log1p(y). Отсюда всё остальное:
    * оптимальная константа            c* = exp(mean z) - 1  (геометрическое среднее);
    * оптимальный прогноз              ŷ*(x) = exp(E[z | x]) - 1;
    * поскольку log1p(0) = 0, точное разложение   E[z | x] = p(x) · m(x).
"""
from __future__ import annotations

import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Корень из среднего квадрата разницы логарифмов. Отрицательное зануляется."""
    p = np.clip(np.asarray(y_pred, dtype="float64"), 0.0, None)
    t = np.asarray(y_true, dtype="float64")
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(t)) ** 2)))


def rmse_log(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    """То же самое, но если обе величины уже в log1p-шкале."""
    return float(np.sqrt(np.mean((np.asarray(z_pred, "float64")
                                  - np.asarray(z_true, "float64")) ** 2)))


def best_constant(y_true: np.ndarray) -> float:
    """Оптимальный константный прогноз под RMSLE."""
    return float(np.expm1(np.log1p(np.asarray(y_true, "float64")).mean()))


def hurdle_glue(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Правило склейки под RMSLE: ŷ = exp(p·m) − 1.

    Это точное, а не приближённое равенство: нулевая компонента вносит в
    E[log1p(y) | x] ровно ноль, потому что log1p(0) = 0.
    """
    p = np.clip(np.asarray(p, "float64"), 0.0, 1.0)
    m = np.clip(np.asarray(m, "float64"), 0.0, None)
    return np.expm1(p * m)


def hurdle_glue_naive(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Интуитивная склейка p · (exp(m) − 1) — оптимальна для MSE в рублях, не для RMSLE.

    Оставлена только для сравнения: на валидации проигрывает примерно 0.37 RMSLE
    при полностью идентичных p и m.
    """
    p = np.clip(np.asarray(p, "float64"), 0.0, 1.0)
    return p * np.expm1(np.clip(np.asarray(m, "float64"), 0.0, None))


def correct_oversampling_prior(p_sampled: np.ndarray, weight: float) -> np.ndarray:
    """Вернуть prior после передискретизации позитивов с коэффициентом `weight`.

    Без этой поправки передискретизация улучшает AUC и ухудшает RMSLE:
    порядок объектов не меняется, а калибровка — которая тут и есть метрика — да.
    """
    p = np.clip(np.asarray(p_sampled, "float64"), 1e-12, 1 - 1e-12)
    return p / (p + (1.0 - p) * weight)


def level_shift_cost(base_rmsle: float, delta: float) -> float:
    """Во что обойдётся постоянное смещение `delta` в log-шкале.

    RMSLE²(смещ) = RMSLE²(база) + delta². При базе около 1.8 смещение входит
    квадратично и потому сильно обесценивается: delta = 0.12 стоит ~0.005.
    """
    return float(np.sqrt(base_rmsle ** 2 + delta ** 2) - base_rmsle)


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Разбор качества прогноза: метрика плюс диагностика калибровки."""
    y_true = np.asarray(y_true, "float64")
    y_pred = np.clip(np.asarray(y_pred, "float64"), 0.0, None)
    zt, zp = np.log1p(y_true), np.log1p(y_pred)
    pos = y_true > 0
    return {
        "rmsle": rmsle(y_true, y_pred),
        "bias_log": float(zp.mean() - zt.mean()),      # >0 — модель завышает
        "mean_log1p_true": float(zt.mean()),
        "mean_log1p_pred": float(zp.mean()),
        "zero_rate_true": float((~pos).mean()),
        "share_pred_nonzero": float((y_pred > 0.5).mean()),
        "rmsle_on_zeros": rmsle(y_true[~pos], y_pred[~pos]) if (~pos).any() else float("nan"),
        "rmsle_on_positive": rmsle(y_true[pos], y_pred[pos]) if pos.any() else float("nan"),
    }
