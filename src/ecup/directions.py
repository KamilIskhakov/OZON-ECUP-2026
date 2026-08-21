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
    # Знак обязателен. G = C²/D положителен всегда, поэтому одного gain
    # недостаточно как условия допуска: когортная карта давала +0.0002
    # на обоих фолдах при корреляциях -0.0169 и +0.0123, то есть
    # оптимальный alpha на них имел противоположные знаки и применить
    # карту с фиксированным знаком было нельзя.
    var_p = float((dperp ** 2).mean())
    C_signed = float((e * dperp).mean())
    return {
        "G_solo": G_solo, "gain_solo": dR_solo,
        "G_marginal": G_marg, "gain_marginal": dR_marg,
        "C_signed": C_signed,
        "alpha_signed": C_signed / var_p if var_p > 1e-15 else 0.0,
        "corr_with_e": float(np.corrcoef(e, d)[0, 1]),
        "orthogonal_share": keep,
        "corr_with_existing": [float(np.corrcoef(d, _center(x))[0, 1]) for x in existing],
    }


def gate(folds: list[dict], min_gain: float = 0.0002) -> tuple[bool, str]:
    """Условие допуска кандидата: величина И совпадение знака между фолдами.

    Ни одно из двух по отдельности не достаточно. Величина без знака
    пропустила бы когортную карту, у которой alpha на соседних окнах
    противоположны. Знак без величины пропустил бы любой слабый шум
    со случайно совпавшим направлением.
    """
    signs = {int(np.sign(f["alpha_signed"])) for f in folds}
    gains = [f["gain_marginal"] for f in folds]
    if len(signs) > 1 or 0 in signs:
        return False, f"знак alpha не переносится: {[round(f['alpha_signed'], 4) for f in folds]}"
    if min(gains) < min_gain:
        return False, f"маржинальный выигрыш ниже порога: {[round(g, 5) for g in gains]}"
    return True, f"знак стабилен, маржинально {[round(g, 5) for g in gains]}"


def report_gate(name: str, folds: list[dict], min_gain: float = 0.0002) -> bool:
    ok, why = gate(folds, min_gain)
    print(f"  {name}: {'ПРОЙДЕН' if ok else 'отклонён'} — {why}")
    return ok


def report(name: str, res: dict) -> None:
    rho = ", ".join(f"{r:+.3f}" for r in res["corr_with_existing"]) or "—"
    print(f"  {name:<30} ρ(e) {res['corr_with_e']:+.4f} · ρ(имеющиеся) [{rho}]")
    print(f"  {'':<30} alpha {res['alpha_signed']:+.4f} · сольно {res['gain_solo']:+.5f} · "
          f"маржинально {res['gain_marginal']:+.5f} · "
          f"ортогональная доля {res['orthogonal_share']:.3f}")
