"""Маржинальный ресурс нового направления после проекции на найденные.

После v18 задача изменилась: нужен не ещё один хороший прогноз, а новый
базисный вектор ошибки. Сольный выигрыш кандидата больше не является
критерием — важно, сколько он объясняет СВЕРХ уже найденных направлений.

Пусть D = [d_1, ..., d_K] — имеющиеся направления. Вычитаем из кандидата
всё, что ими объясняется:

    d^perp = d - D (D'D)^{-1} D' d

и считаем ресурс ортогональной части:

    G = Cov(e, d^perp)^2 / Var(d^perp)

Это ровно та часть MSE, которую новый механизм снимает дополнительно.
Величина G в шкале MSE; в шкале RMSLE выигрыш равен sqrt(M) - sqrt(M - G).
"""
from __future__ import annotations

import numpy as np


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def orthogonalize(d: np.ndarray, existing: list[np.ndarray]) -> np.ndarray:
    """Остаток кандидата после проекции на подпространство найденных."""
    d = _center(np.asarray(d, dtype="float64"))
    if not existing:
        return d
    D = np.column_stack([_center(np.asarray(x, dtype="float64")) for x in existing])
    coef, *_ = np.linalg.lstsq(D, d, rcond=None)
    return d - D @ coef


def marginal_gain(e: np.ndarray, d: np.ndarray,
                  existing: list[np.ndarray] | None = None,
                  base_mse: float | None = None) -> dict:
    """Сольный и маржинальный ресурс кандидата.

    `e` — остаток текущей базы, `d` — направление кандидата,
    `existing` — уже найденные направления.
    """
    e = _center(np.asarray(e, dtype="float64"))
    d = _center(np.asarray(d, dtype="float64"))
    existing = existing or []
    M = float((e ** 2).mean()) if base_mse is None else base_mse

    def resource(v):
        var = float((v ** 2).mean())
        if var < 1e-15:
            return 0.0, 0.0
        cov = float((e * v).mean())
        G = cov * cov / var
        return G, np.sqrt(M) - np.sqrt(max(M - G, 0.0))

    G_solo, dR_solo = resource(d)
    dperp = orthogonalize(d, existing)
    G_marg, dR_marg = resource(dperp)
    keep = float(np.sqrt((dperp ** 2).mean() / max((d ** 2).mean(), 1e-15)))
    return {
        "G_solo": G_solo, "gain_solo": dR_solo,
        "G_marginal": G_marg, "gain_marginal": dR_marg,
        "corr_with_e": float(np.corrcoef(e, d)[0, 1]),
        "orthogonal_share": keep,
        "corr_with_existing": [float(np.corrcoef(d, _center(x))[0, 1]) for x in existing],
    }


def report(name: str, res: dict) -> None:
    rho = ", ".join(f"{r:+.3f}" for r in res["corr_with_existing"]) or "—"
    print(f"  {name:<30} ρ(e) {res['corr_with_e']:+.4f} · ρ(имеющиеся) [{rho}]")
    print(f"  {'':<30} сольно {res['gain_solo']:+.5f} · "
          f"маржинально {res['gain_marginal']:+.5f} · "
          f"ортогональная доля {res['orthogonal_share']:.3f}")
