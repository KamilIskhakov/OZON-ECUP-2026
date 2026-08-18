"""Hurdle на CatBoost — участник ансамбля с другой геометрией разбиений.

Зачем отдельное семейство, а не ещё один сид LightGBM. Все реальные приросты
на лидерборде дал ансамбль, и его выигрыш определяется не качеством участников,
а корреляцией их ошибок: при равных дисперсиях смесь K моделей с попарной
корреляцией ρ имеет дисперсию ошибки σ²(ρ + (1-ρ)/K). Сид и ёмкость двигают ρ
слабо, потому что алгоритм роста дерева один и тот же — жадный листовой рост
по одному и тому же критерию. CatBoost меняет сам алгоритм:

* **симметричные (oblivious) деревья** — на каждом уровне одно и то же разбиение
  для всех узлов. Это сильная структурная регуляризация: модель не может
  выделить узкий лист под конкретную подгруппу, зато её ошибки распределены
  иначе, чем у листового роста LightGBM;
* **упорядоченный бустинг** — градиент для объекта считается моделью, обученной
  без этого объекта, что убирает смещение оценки градиента (target leakage
  обычного бустинга). На разреженных признаках вроде recency это заметно;
* **другая дискретизация признаков** — border_count квантилей вместо
  гистограмм LightGBM, то есть другие точки разбиения на тех же данных.

Семантика офсетов совпадает с `HurdleGBDT` дословно, иначе участники нельзя
складывать: классификатор получает logit базовой доли покупателей якоря
как `baseline` (CatBoost учит приращение к нему и на предсказании возвращает
только приращение — ровно как init_score в LightGBM), регрессия обучается
на отклонении z от условного уровня якоря ℓ⁺.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .metrics import hurdle_glue


@dataclass
class CatBoostConfig:
    """Гиперпараметры, подобранные как аналог консервативной LightGBM-конфигурации.

    `depth=6` даёт 2⁶ = 64 листа — сопоставимо с `num_leaves=63`, но у CatBoost
    это полное симметричное дерево, а не 63 листа произвольной формы.
    """

    clf_params: dict = field(default_factory=lambda: dict(
        loss_function="Logloss", learning_rate=0.05, depth=6,
        l2_leaf_reg=5.0, min_data_in_leaf=200, iterations=600,
        rsm=0.8, bootstrap_type="Bernoulli", subsample=0.8,
        border_count=128, verbose=0, allow_writing_files=False, thread_count=-1,
    ))
    reg_params: dict = field(default_factory=lambda: dict(
        loss_function="RMSE", learning_rate=0.05, depth=6,
        l2_leaf_reg=5.0, min_data_in_leaf=200, iterations=600,
        rsm=0.8, bootstrap_type="Bernoulli", subsample=0.8,
        border_count=128, verbose=0, allow_writing_files=False, thread_count=-1,
    ))
    seed: int = 42
    early_stopping_rounds: int | None = 100
    eval_frac: float = 0.12
    refit_full: bool = False


@dataclass
class HurdleCatBoost:
    """P(y>0|x) и E[log1p(y)|x, y>0] на CatBoost. API совпадает с `HurdleGBDT`."""

    config: CatBoostConfig = field(default_factory=CatBoostConfig)
    feature_names: list[str] | None = None
    clf: Any = None
    reg: Any = None
    _clf_has_init: bool = False
    _refit_iters: tuple[int, int] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
        z_offset: np.ndarray | None = None,
        clf_init: np.ndarray | None = None,
        early_stopping_rounds: int | None = None,
        eval_frac: float | None = None,
        refit_full: bool | None = None,
    ) -> "HurdleCatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        cfg = self.config
        esr = cfg.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds
        eval_frac = cfg.eval_frac if eval_frac is None else eval_frac
        refit_full = cfg.refit_full if refit_full is None else refit_full

        y = np.asarray(y, dtype="float64")
        pos = y > 0
        self.feature_names = feature_names or self.feature_names
        init = None if clf_init is None else np.asarray(clf_init, "float64")
        self._clf_has_init = clf_init is not None

        if esr:
            idx_tr, idx_es = train_test_split(
                np.arange(len(y)), test_size=eval_frac,
                random_state=cfg.seed, stratify=pos)
        else:
            idx_tr, idx_es = np.arange(len(y)), None

        sub = lambda a, i: None if a is None else a[i]

        def pool(i, label, base=None):
            return Pool(X[i], label=label[i], weight=sub(sample_weight, i),
                        baseline=sub(base, i))

        lab_c = pos.astype(np.int8)
        self.clf = CatBoostClassifier(random_seed=cfg.seed, **cfg.clf_params)
        self.clf.fit(pool(idx_tr, lab_c, init),
                     eval_set=None if idx_es is None else pool(idx_es, lab_c, init),
                     early_stopping_rounds=esr or None, verbose=False)

        if pos.sum() < 100:
            raise ValueError(f"позитивов слишком мало для регрессии: {int(pos.sum())}")
        z = np.log1p(y)
        if z_offset is not None:
            z = z - np.asarray(z_offset, dtype="float64")
        ptr = idx_tr[pos[idx_tr]]
        self.reg = CatBoostRegressor(random_seed=cfg.seed, **cfg.reg_params)
        self.reg.fit(pool(ptr, z),
                     eval_set=None if idx_es is None else pool(idx_es[pos[idx_es]], z),
                     early_stopping_rounds=esr or None, verbose=False)

        # То же соображение, что и у LightGBM-ветки: число деревьев уже найдено
        # на отложенной доле, и её можно вернуть в обучение, домножив оптимум
        # на 1/(1-eval_frac).
        if refit_full and idx_es is not None:
            scale = 1.0 / (1.0 - eval_frac)
            bc, br = self.best_iters
            cp = {**cfg.clf_params, "iterations": max(1, int(round(bc * scale)))}
            rp = {**cfg.reg_params, "iterations": max(1, int(round(br * scale)))}
            self.clf = CatBoostClassifier(random_seed=cfg.seed, **cp)
            self.clf.fit(Pool(X, label=lab_c, weight=sample_weight, baseline=init),
                         verbose=False)
            self.reg = CatBoostRegressor(random_seed=cfg.seed, **rp)
            self.reg.fit(Pool(X[pos], label=z[pos], weight=sub(sample_weight, pos)),
                         verbose=False)
            self._refit_iters = (cp["iterations"], rp["iterations"])
        return self

    @property
    def best_iters(self) -> tuple[int, int]:
        def n(model):
            bi = model.get_best_iteration()
            return (bi + 1) if bi is not None else model.tree_count_
        return n(self.clf), n(self.reg)

    def predict_parts(
        self,
        X: np.ndarray,
        p_target: float | None = None,
        m_offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(p, m) для целевого окна — семантика ручек как у `HurdleGBDT`.

        Обучение с `baseline` делает модель оценщиком ПРИРАЩЕНИЯ к логиту
        базовой доли, и `RawFormulaVal` возвращает именно приращение. Поэтому
        подстановка логита целевого окна здесь — не эвристика, а восстановление
        того же уровня, относительно которого модель училась.
        """
        raw = self.clf.predict(X, prediction_type="RawFormulaVal")
        init = 0.0 if p_target is None else float(np.log(p_target / (1.0 - p_target)))
        p = 1.0 / (1.0 + np.exp(-(raw + init)))
        return p, self.reg.predict(X) + m_offset

    def predict(self, X: np.ndarray, p_target: float | None = None,
                m_offset: float = 0.0) -> np.ndarray:
        p, m = self.predict_parts(X, p_target=p_target, m_offset=m_offset)
        return hurdle_glue(p, np.clip(m, 0.0, None))

    def importances(self, top: int = 30) -> list[tuple[str, float, float]]:
        names = self.feature_names or [f"f{i}" for i in range(self.clf.tree_count_)]
        ic = np.asarray(self.clf.get_feature_importance(), dtype="float64")
        ir = np.asarray(self.reg.get_feature_importance(), dtype="float64")
        ic, ir = ic / max(ic.sum(), 1e-12), ir / max(ir.sum(), 1e-12)
        order = np.argsort(-(ic + ir))[:top]
        return [(names[i], float(ic[i]), float(ir[i])) for i in order]
