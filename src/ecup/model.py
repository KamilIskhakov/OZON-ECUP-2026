"""Модели: двухчастная hurdle и прямая регрессия для сравнения.

Важно понимать, чем они отличаются на самом деле. GBDT с MSE-лоссом на таргете
z = log1p(y) уже оценивает E[z | x] = p(x)·m(x), то есть прямая модель — это не
наивный бейзлайн, а теоретически корректный оценщик. Hurdle выигрывает только
если p и m требуют разной ёмкости или разных признаков; проверять это надо
сравнением на одинаковых временных фолдах, а не по доле нулей.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from .config import ModelConfig
from .metrics import hurdle_glue


@dataclass
class HurdleGBDT:
    """P(y>0 | x) классификатором, E[log1p(y) | x, y>0] регрессией на позитивах."""

    config: ModelConfig = field(default_factory=ModelConfig)
    feature_names: list[str] | None = None
    clf: lgb.LGBMClassifier | None = None
    reg: lgb.LGBMRegressor | None = None
    _clf_has_init: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
        z_offset: np.ndarray | None = None,
        clf_init: np.ndarray | None = None,
    ) -> "HurdleGBDT":
        """Обучить обе головы, каждую со своим офсетом якоря.

        `clf_init` — logit базовой доли покупателей якоря, подаётся как
        init_score: классификатор учит отклонение от неё, а не абсолютный
        уровень. `z_offset` — уровень якоря для регрессии; вычитать надо
        УСЛОВНОЕ среднее ℓ⁺ (по покупавшим), а не безусловное ℓ: регрессия
        обучается только на y>0, а ℓ = p̄·ℓ⁺ почти целиком описывает
        экстенсивную маржу и внёс бы в цель посторонний уровневый шум.
        """
        y = np.asarray(y, dtype="float64")
        pos = y > 0
        self.feature_names = feature_names or self.feature_names

        self.clf = lgb.LGBMClassifier(random_state=self.config.seed, **self.config.clf_params)
        self.clf.fit(X, pos.astype(np.int8), sample_weight=sample_weight,
                     init_score=None if clf_init is None else np.asarray(clf_init, "float64"))
        self._clf_has_init = clf_init is not None

        if pos.sum() < 100:
            raise ValueError(f"позитивов слишком мало для регрессии: {int(pos.sum())}")
        w_pos = None if sample_weight is None else sample_weight[pos]
        z = np.log1p(y[pos])
        if z_offset is not None:
            z = z - np.asarray(z_offset, dtype="float64")[pos]
        self.reg = lgb.LGBMRegressor(random_state=self.config.seed, **self.config.reg_params)
        self.reg.fit(X[pos], z, sample_weight=w_pos)
        return self

    def predict_parts(
        self,
        X: np.ndarray,
        p_target: float | None = None,
        m_offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(p, m) для целевого окна.

        `p_target` — желаемая базовая доля покупателей: прибавляется к логиту,
        то есть умножает шансы. Так сезонность попадает в экстенсивную маржу,
        где она и живёт: сильнее всего сдвигаются пользователи с p около 0.5,
        а те, кто и так купит с вероятностью 0.99, почти не двигаются.

        Регрессия без обрезки снизу: с z_offset она предсказывает отклонение
        от уровня якоря, и отрицательные значения там законны.
        """
        if self._clf_has_init:
            raw = self.clf.booster_.predict(X, raw_score=True)
            init = 0.0 if p_target is None else float(np.log(p_target / (1.0 - p_target)))
            p = 1.0 / (1.0 + np.exp(-(raw + init)))
        else:
            p = self.clf.predict_proba(X)[:, 1]
            if p_target is not None:
                lo = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
                p = 1.0 / (1.0 + np.exp(-(lo + float(np.log(p_target / (1 - p_target))))))
        return p, self.reg.predict(X) + m_offset

    def predict(
        self,
        X: np.ndarray,
        p_target: float | None = None,
        m_offset: float = 0.0,
    ) -> np.ndarray:
        """Прогноз в рублях. Уровень задаётся двумя ручками, а не одним скаляром.

        Экстенсивная часть — через `p_target`, интенсивная — через `m_offset`.
        Раньше здесь был один сдвиг, прибавляемый к m: он давал каждому
        пользователю сдвиг p_i·delta, то есть распределял поправку ровно
        против механизма праздничного спроса.
        """
        p, m = self.predict_parts(X, p_target=p_target, m_offset=m_offset)
        return hurdle_glue(p, np.clip(m, 0.0, None))

    def importances(self, top: int = 30) -> list[tuple[str, float, float]]:
        """(признак, важность в классификаторе, важность в регрессии), нормировано."""
        names = self.feature_names or [f"f{i}" for i in range(len(self.clf.feature_importances_))]
        ic = self.clf.booster_.feature_importance("gain").astype("float64")
        ir = self.reg.booster_.feature_importance("gain").astype("float64")
        ic = ic / max(ic.sum(), 1)
        ir = ir / max(ir.sum(), 1)
        order = np.argsort(-(ic + ir))[:top]
        return [(names[i], float(ic[i]), float(ir[i])) for i in order]


@dataclass
class DirectGBDT:
    """Одна регрессия на log1p(y) с MSE. Референс, с которым сравнивается hurdle."""

    config: ModelConfig = field(default_factory=ModelConfig)
    feature_names: list[str] | None = None
    reg: lgb.LGBMRegressor | None = None

    def fit(self, X, y, feature_names=None, sample_weight=None,
            z_offset=None) -> "DirectGBDT":
        self.feature_names = feature_names or self.feature_names
        z = np.log1p(np.asarray(y, "float64"))
        if z_offset is not None:
            z = z - np.asarray(z_offset, dtype="float64")
        self.reg = lgb.LGBMRegressor(random_state=self.config.seed, **self.config.reg_params)
        self.reg.fit(X, z, sample_weight=sample_weight)
        return self

    def predict(self, X, level_shift: float = 0.0) -> np.ndarray:
        z = np.clip(self.reg.predict(X) + level_shift, 0.0, None)
        return np.expm1(z)

    def importances(self, top: int = 30) -> list[tuple[str, float]]:
        names = self.feature_names or [f"f{i}" for i in range(len(self.reg.feature_importances_))]
        imp = self.reg.booster_.feature_importance("gain").astype("float64")
        imp = imp / max(imp.sum(), 1)
        order = np.argsort(-imp)[:top]
        return [(names[i], float(imp[i])) for i in order]
