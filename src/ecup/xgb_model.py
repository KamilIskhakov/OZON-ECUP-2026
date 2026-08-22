"""XGBoost третьим древесным семейством — hurdle и прямая регрессия.

API повторяет `HurdleCatBoost`, чтобы обвязка не менялась. Аналог
`init_score` LightGBM и `baseline` CatBoost здесь — `base_margin`.
Чтобы сырой выход был именно ПРИРАЩЕНИЕМ к переданному уровню, а не
приращением к внутреннему `base_score`, последний зануляется явно.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .metrics import hurdle_glue

_COMMON = dict(tree_method="hist", max_depth=7, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
               reg_lambda=5.0, n_estimators=600, verbosity=0)

# base_score задаётся в шкале ОТКЛИКА, а не логита: для binary:logistic
# это вероятность, и 0.5 даёт нулевой логит, поэтому output_margin
# возвращает чистое приращение к переданному base_margin. Ноль здесь
# означал бы логит -inf и ломает классификатор.
_CLF = dict(_COMMON, objective="binary:logistic", base_score=0.5)
_REG = dict(_COMMON, objective="reg:squarederror", base_score=0.0)


@dataclass
class XGBConfig:
    clf_params: dict = field(default_factory=lambda: dict(_CLF))
    reg_params: dict = field(default_factory=lambda: dict(_REG))
    seed: int = 42
    device: str = "cpu"


@dataclass
class HurdleXGB:
    config: XGBConfig = field(default_factory=XGBConfig)
    feature_names: list[str] | None = None
    clf: Any = None
    reg: Any = None

    def fit(self, X, y, feature_names=None, sample_weight=None,
            z_offset=None, clf_init=None):
        import xgboost as xgb
        cfg = self.config
        y = np.asarray(y, dtype="float64"); pos = y > 0
        self.feature_names = feature_names or self.feature_names
        init = None if clf_init is None else np.asarray(clf_init, "float64")

        self.clf = xgb.XGBClassifier(random_state=cfg.seed, device=cfg.device,
                                     **cfg.clf_params)
        self.clf.fit(X, pos.astype(np.int8), sample_weight=sample_weight,
                     base_margin=init)

        z = np.log1p(y)
        if z_offset is not None:
            z = z - np.asarray(z_offset, dtype="float64")
        w = None if sample_weight is None else sample_weight[pos]
        self.reg = xgb.XGBRegressor(random_state=cfg.seed, device=cfg.device,
                                    **cfg.reg_params)
        self.reg.fit(X[pos], z[pos], sample_weight=w)
        return self

    def predict_parts(self, X, p_target=None, m_offset=0.0):
        raw = self.clf.predict(X, output_margin=True)
        init = 0.0 if p_target is None else float(np.log(p_target / (1.0 - p_target)))
        p = 1.0 / (1.0 + np.exp(-(raw + init)))
        return p, self.reg.predict(X) + m_offset

    def predict(self, X, p_target=None, m_offset=0.0):
        p, m = self.predict_parts(X, p_target=p_target, m_offset=m_offset)
        return hurdle_glue(p, np.clip(m, 0.0, None))


@dataclass
class DirectXGB:
    """Прямая регрессия z без разделения на маржи.

    Уровень якоря вычитается БЕЗУСЛОВНЫЙ: модель учится на всех строках,
    включая нулевые, и цель непокупателя должна оставаться нулём минус
    общий уровень, а не минус условное среднее по покупавшим.
    """
    config: XGBConfig = field(default_factory=XGBConfig)
    reg: Any = None

    def fit(self, X, y, feature_names=None, sample_weight=None, z_offset=None):
        import xgboost as xgb
        cfg = self.config
        z = np.log1p(np.asarray(y, dtype="float64"))
        if z_offset is not None:
            z = z - np.asarray(z_offset, dtype="float64")
        self.reg = xgb.XGBRegressor(random_state=cfg.seed, device=cfg.device,
                                    **cfg.reg_params)
        self.reg.fit(X, z, sample_weight=sample_weight)
        return self

    def predict(self, X, level=0.0):
        return np.expm1(np.clip(self.reg.predict(X) + level, 0.0, None))
