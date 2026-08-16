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
from sklearn.model_selection import train_test_split

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
        eval_frac: float = 0.12,
        refit_full: bool = False,
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
        n = len(y)
        init = None if clf_init is None else np.asarray(clf_init, "float64")
        self._clf_has_init = clf_init is not None

        # Отложенная часть для ранней остановки. Делим по пользователям внутри
        # трейна: число деревьев подбирается на не виденных строках, а качество
        # всё равно меряется на отдельном по времени якоре.
        if early_stopping_rounds:
            idx_tr, idx_es = train_test_split(
                np.arange(n), test_size=eval_frac, random_state=self.config.seed,
                stratify=pos)
        else:
            idx_tr, idx_es = np.arange(n), None

        sub = lambda a, i: None if a is None else a[i]
        self.clf = lgb.LGBMClassifier(random_state=self.config.seed, **self.config.clf_params)
        kw = {}
        if idx_es is not None:
            kw = dict(
                eval_set=[(X[idx_es], pos[idx_es].astype(np.int8))],
                eval_metric="binary_logloss",
                eval_sample_weight=None if sample_weight is None else [sample_weight[idx_es]],
                eval_init_score=None if init is None else [init[idx_es]],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
        self.clf.fit(X[idx_tr], pos[idx_tr].astype(np.int8),
                     sample_weight=sub(sample_weight, idx_tr),
                     init_score=sub(init, idx_tr), **kw)

        if pos.sum() < 100:
            raise ValueError(f"позитивов слишком мало для регрессии: {int(pos.sum())}")
        z = np.log1p(y)
        if z_offset is not None:
            z = z - np.asarray(z_offset, dtype="float64")
        ptr = idx_tr[pos[idx_tr]]
        self.reg = lgb.LGBMRegressor(random_state=self.config.seed, **self.config.reg_params)
        kw = {}
        if idx_es is not None:
            pes = idx_es[pos[idx_es]]
            kw = dict(eval_set=[(X[pes], z[pes])], eval_metric="l2",
                      eval_sample_weight=None if sample_weight is None else [sample_weight[pes]],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        self.reg.fit(X[ptr], z[ptr], sample_weight=sub(sample_weight, ptr), **kw)

        # Ранняя остановка отъедает eval_frac обучающих строк. Для финальной
        # модели это чистая потеря: число деревьев уже найдено, и его можно
        # переиспользовать, дообучившись на всех данных. Масштабируем на
        # 1/(1-eval_frac) — с большим трейном оптимум сдвигается вправо.
        if refit_full and idx_es is not None:
            scale = 1.0 / (1.0 - eval_frac)
            bc, br = self.best_iters
            cp = {**self.config.clf_params, "n_estimators": max(1, int(round(bc * scale)))}
            rp = {**self.config.reg_params, "n_estimators": max(1, int(round(br * scale)))}
            self.clf = lgb.LGBMClassifier(random_state=self.config.seed, **cp)
            self.clf.fit(X, pos.astype(np.int8), sample_weight=sample_weight, init_score=init)
            self.reg = lgb.LGBMRegressor(random_state=self.config.seed, **rp)
            self.reg.fit(X[pos], z[pos], sample_weight=sub(sample_weight, pos))
            self._refit_iters = (cp["n_estimators"], rp["n_estimators"])
        return self

    @property
    def best_iters(self) -> tuple[int, int]:
        """Сколько деревьев реально понадобилось каждой голове."""
        return (getattr(self.clf, "best_iteration_", None) or self.clf.n_estimators,
                getattr(self.reg, "best_iteration_", None) or self.reg.n_estimators)

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


def metric_aligned_objective(p: np.ndarray, offset: np.ndarray | None = None):
    """Лосс регрессии, согласованный с итоговой метрикой.

    Текущая схема обучает две головы двумя локальными лоссами: классификатор
    логлоссом, регрессию — MSE на позитивах. Итоговую ошибку (p·m − z)²
    не оптимизирует ни одна из них. Асимптотически это неважно, но при
    конечной выборке, регуляризации и ограниченной ёмкости произведение
    двух состоятельных оценок не обязано быть лучшей оценкой z.

    Здесь регрессия учится прямо под метрику. Если финальный прогноз
    ẑ = p·(f + ℓ), то минимизируется

        L_i(f) = w_i · (p_i·(f_i + ℓ_i) − z_i)²

    Градиент 2·w·p·(p(f+ℓ) − z), гессиан 2·w·p² ≥ 0 — задача выпуклая.

    Уровень ℓ входит ВНУТРЬ скобки, до умножения на p. Вычитать его из
    метки нельзя: тогда лосс стал бы (p·f + ℓ − z)² вместо (p·f + p·ℓ − z)²,
    и ошибка составила бы (1−p)·ℓ — тем большая, чем реже человек покупает.
    Это тот же класс ошибки, что и прибавление сдвига к готовому p·m.

    Веса применяются ВРУЧНУЮ: LightGBM не передаёт sample_weight в custom
    objective с двумя аргументами и не применяет их к возвращённым
    градиентам — проверено эмпирически. Поэтому сигнатура трёхаргументная.

    Что оценивает f. При истинном p оптимум f* = E[z|x]/p(x) = m(x). Но если
    классификатор смещён, оптимум становится p(x)m(x)/p̃(x), то есть f
    начинает компенсировать ошибку классификатора. Для метрики это хорошо
    (произведение p̃·f по-прежнему стремится к E[z|x]), но интерпретация
    «условная сумма покупки» теряется — поэтому величина называется
    conditional score, а не m.

    `p` обязан быть OOF-прогнозом: иначе в цель просачивается информация
    о таргете тех же строк.
    """
    p = np.asarray(p, dtype="float64")
    off = None if offset is None else np.asarray(offset, dtype="float64")

    def objective(y_true, y_pred, weight):
        f = y_pred if off is None else y_pred + off
        resid = p * f - y_true
        grad = 2.0 * p * resid
        hess = 2.0 * p * p
        if weight is not None:
            grad = grad * weight
            hess = hess * weight
        return grad, hess

    return objective


def metric_aligned_init(p, z, offset=None, weight=None) -> float:
    """Аналитический стартовый скор для этого лосса.

    Оптимальная константа c решает d/dc Σ w p (p(c+ℓ) − z)² = 0:

        c* = Σ w p (z − p ℓ) / Σ w p²

    Медиана положительных, которую легко подставить по привычке, к оптимуму
    этого лосса отношения не имеет.
    """
    p = np.asarray(p, "float64"); z = np.asarray(z, "float64")
    w = np.ones_like(p) if weight is None else np.asarray(weight, "float64")
    num = w * p * (z if offset is None else z - p * np.asarray(offset, "float64"))
    return float(num.sum() / max((w * p * p).sum(), 1e-12))


@dataclass
class MetricAlignedGBDT:
    """Conditional score под лосс (p·(f+ℓ) − z)² при зафиксированном OOF p."""

    config: ModelConfig = field(default_factory=ModelConfig)
    feature_names: list[str] | None = None
    reg: lgb.LGBMRegressor | None = None
    _init: float = 0.0

    def fit(self, X, y, p_oof, feature_names=None, sample_weight=None,
            m_offset=None) -> "MetricAlignedGBDT":
        """`m_offset` — уровень ℓ⁺ якоря; None означает абсолютный режим."""
        self.feature_names = feature_names or self.feature_names
        z = np.log1p(np.asarray(y, "float64"))
        self._init = metric_aligned_init(p_oof, z, m_offset, sample_weight)
        params = {k: v for k, v in self.config.reg_params.items()
                  if k not in ("objective", "metric")}
        # Стартовый скор обязан входить В ОБУЧЕНИЕ, а не только в predict:
        # иначе бустинг ищет f с p·f ≈ z, а наружу отдаётся f + init.
        base = np.full(len(z), self._init, dtype="float64")
        off = base if m_offset is None else np.asarray(m_offset, "float64") + base
        self.reg = lgb.LGBMRegressor(
            random_state=self.config.seed,
            objective=metric_aligned_objective(p_oof, off), **params)
        # init_score не входит в predict — проверено; прибавляем вручную
        self.reg.fit(X, z, sample_weight=sample_weight,
                     init_score=np.zeros(len(z)))
        return self

    def predict(self, X, m_offset: float = 0.0) -> np.ndarray:
        """Conditional score; умножать на p для итогового ẑ."""
        return self.reg.predict(X) + self._init + m_offset
