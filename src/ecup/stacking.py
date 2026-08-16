"""Стекинг: веса ансамбля как решение задачи оптимизации, а не подбор.

Равные веса оптимальны только если все участники имеют одинаковую дисперсию
ошибки и одинаковые попарные корреляции. У нас это не так: direct-модели
заметно слабее hurdle поодиночке, а модели с разной глубиной окна коррелируют
между собой сильнее, чем с моделями другого типа.

Постановка. Пусть $z$ — истинный log1p таргета, $\\hat z_k$ — прогноз k-й модели.
Ищем веса

    min_w  || Σ_k w_k ẑ_k − z ||²     при     w ≥ 0,  Σ_k w_k = 1

Неотрицательность и нормировка не косметика: без них решение уходит
в экстраполяцию с огромными разнонаправленными весами, которая идеально
описывает валидацию и разваливается на тесте. Сумма единица дополнительно
гарантирует, что смесь не сдвигает уровень.

Регуляризация к равномерному вектору с силой λ:

    min_w  || Σ_k w_k ẑ_k − z ||² + λ n || w − 1/K ||²

При λ → ∞ решение вырождается в равные веса, то есть в проверенный ансамбль;
λ выбирается по отложенному якорю, а не назначается.

Важно: подгонка идёт по ЦЕНТРИРОВАННЫМ невязкам. Уровень калибруется отдельно
и точно (см. разложение на маржи), поэтому веса должны отвечать за форму
прогноза, иначе они потратятся на компенсацию уровневых смещений участников.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Проекция на симплекс {w ≥ 0, Σw = 1} — алгоритм Duchi et al."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def fit_weights(
    Z: np.ndarray,
    z_true: np.ndarray,
    lam: float = 0.0,
    iters: int = 3000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Веса на симплексе, минимизирующие MSE центрированной невязки.

    Проекционный градиентный спуск: задача выпуклая (квадратичная на выпуклом
    множестве), поэтому сходимость к глобальному минимуму гарантирована,
    а проекция на симплекс делается за O(K log K).

    `Z` — (K, n) прогнозы участников в log1p-шкале, `z_true` — (n,).
    """
    K = Z.shape[0]
    Zc = Z - Z.mean(axis=1, keepdims=True)      # центрируем: веса отвечают за форму
    tc = z_true - z_true.mean()
    G = (Zc @ Zc.T) / Zc.shape[1]
    b = (Zc @ tc) / Zc.shape[1]
    uni = np.full(K, 1.0 / K)
    if lam > 0:                                  # ридж к равным весам
        G = G + lam * np.eye(K)
        b = b + lam * uni
    step = 1.0 / (np.linalg.eigvalsh(G).max() + 1e-12)
    w = uni.copy()
    for _ in range(iters):
        w_new = _project_simplex(w - step * (G @ w - b))
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    return w


def shape_of(z_pred: np.ndarray, z_true: np.ndarray) -> float:
    """std невязки — та часть ошибки, которую не чинит сдвиг уровня."""
    e = z_true - z_pred
    return float(e.std())


@dataclass
class StackResult:
    weights: np.ndarray
    lam: float
    shape_stack: float
    shape_uniform: float
    shape_best_single: int

    @property
    def gain_over_uniform(self) -> float:
        return self.shape_uniform - self.shape_stack


def select_lambda(
    Z_fit: np.ndarray,
    z_fit: np.ndarray,
    Z_val: np.ndarray,
    z_val: np.ndarray,
    lambdas: tuple[float, ...] = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0),
) -> StackResult:
    """Подобрать силу регуляризации на отложенном якоре.

    Веса учатся на одном якоре, проверяются на другом. Если ни одна λ
    не бьёт равные веса на отложенном якоре — значит стекинг здесь
    не окупается, и это честный отрицательный ответ, а не повод
    крутить дальше.
    """
    uni = np.full(Z_val.shape[0], 1.0 / Z_val.shape[0])
    s_uni = shape_of(uni @ Z_val, z_val)
    best = None
    for lam in lambdas:
        w = fit_weights(Z_fit, z_fit, lam=lam)
        s = shape_of(w @ Z_val, z_val)
        if best is None or s < best[1]:
            best = (w, s, lam)
    singles = [shape_of(Z_val[k], z_val) for k in range(Z_val.shape[0])]
    return StackResult(weights=best[0], lam=best[2], shape_stack=best[1],
                       shape_uniform=s_uni, shape_best_single=int(np.argmin(singles)))


def fit_isotonic_calibration(z_pred: np.ndarray, z_true: np.ndarray):
    """Монотонная перекалибровка итогового оценщика.

    Тождество E[z|x] = p(x)·m(x) точно для ИСТИННЫХ p и m. Но оцениваем мы их
    разными лоссами: классификатор логлоссом, регрессию — MSE на позитивах.
    Ни один из двух не видел итоговой ошибки (p̂m̂ − z)², поэтому произведение
    состоятельно, но в конечной выборке не оптимизировано под метрику.

    Изотоническая регрессия ищет монотонную g, минимизирующую Σ(g(ẑ) − z)².
    Монотонность — единственное ограничение, и оно содержательное: калибровка
    не должна менять порядок пользователей, только шкалу. Гиперпараметров нет,
    алгоритм PAVA работает за O(n log n).

    Подгонка идёт по ЦЕНТРИРОВАННЫМ величинам: уровень окна калибруется
    отдельно и точно, а между якорями он разный — иначе g выучила бы сдвиг
    одного якоря и перенесла его на другой.
    """
    from sklearn.isotonic import IsotonicRegression

    mu_p, mu_t = float(z_pred.mean()), float(z_true.mean())
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(z_pred - mu_p, z_true - mu_t)

    def apply(z: np.ndarray, target_level: float | None = None) -> np.ndarray:
        out = iso.predict(z - z.mean())
        return out + (z.mean() if target_level is None else target_level)

    return apply


def evaluate_calibration(z_fit, y_fit, z_val, y_val) -> dict:
    """Обучить калибровку на одном якоре, проверить на другом.

    Если на отложенном якоре она не улучшает shape — значит перекалибровка
    здесь не нужна, и это честный ответ, а не повод менять схему подгонки.
    """
    zt_fit, zt_val = np.log1p(y_fit), np.log1p(y_val)
    g = fit_isotonic_calibration(z_fit, zt_fit)
    before = shape_of(z_val, zt_val)
    after = shape_of(g(z_val), zt_val)
    return {"shape_before": before, "shape_after": after,
            "gain": before - after, "apply": g}
