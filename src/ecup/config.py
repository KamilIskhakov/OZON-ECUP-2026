"""Константы задачи и пути. Всё, что зависит от данных соревнования, живёт здесь."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path


def _find_root(start: Path | None = None) -> Path:
    """Корень проекта — ближайшая вверх по дереву папка, содержащая data_start/."""
    p = (start or Path(__file__)).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data_start").exists():
            return cand
    raise FileNotFoundError("не найдена папка data_start/ вверх по дереву каталогов")


ROOT = _find_root()
DATA_START = ROOT / "data_start"
DATA_WORK = ROOT / "data_work"
ARTIFACTS = ROOT / "artifacts"

TRAIN_PARQUET = DATA_START / "train.parquet"
SAMPLE_SUBMIT = DATA_START / "sample_submit.csv"
COMPACT_PARQUET = DATA_WORK / "train_compact.parquet"

# --- календарь ---------------------------------------------------------------
DAY0 = date(2025, 1, 1)          # d = 0
DAY_LAST = date(2026, 2, 13)     # d = 408, последний наблюдаемый день
N_DAYS = (DAY_LAST - DAY0).days + 1
HORIZON = 30                     # длина таргет-окна, дней

ANCHOR_FINAL = N_DAYS - 1              # 408 — боевой якорь, таргет невидим
ANCHOR_VAL = ANCHOR_FINAL - HORIZON    # 378 — последний якорь с полным таргетом

N_USERS_EXPECTED = 250_000

# --- правило отбора пользователей (восстановлено в 01_eda.ipynb §3.1) ---------
# В выборку вошли те, у кого есть событие в каждом из трёх последних
# 30-дневных блоков. На боевом якоре правилу удовлетворяют ровно все 250 000.
SELECTION_BLOCKS = 3
SELECTION_BLOCK_LEN = 30
SELECTION_SPAN = SELECTION_BLOCKS * SELECTION_BLOCK_LEN   # 90 дней


def d_to_date(d: int) -> date:
    return DAY0 + timedelta(days=int(d))


def date_to_d(dt: date) -> int:
    return (dt - DAY0).days


def anchor_label(d: int) -> str:
    return d_to_date(d).isoformat()


def target_window(anchor: int, horizon: int = HORIZON) -> tuple[date, date]:
    return d_to_date(anchor + 1), d_to_date(anchor + horizon)


@dataclass(frozen=True)
class Windows:
    """Длины оконных агрегаций (дни) и лаговых окон [lo, hi] назад от якоря."""

    rolling: tuple[int, ...] = (7, 14, 30, 60, 90, 180, 365)
    lagged: tuple[tuple[int, int], ...] = ((30, 60), (60, 90), (90, 180))

    def clipped(self, max_history: int) -> "Windows":
        """Обрезать под доступную глубину истории.

        Окно длиннее max_history считалось бы по неполным данным и молча
        превращалось бы в дубликат самого длинного допустимого окна.
        """
        roll = tuple(w for w in self.rolling if w <= max_history)
        if not roll:
            roll = (min(self.rolling[0], max_history),)
        lag = tuple((lo, hi) for lo, hi in self.lagged if hi <= max_history)
        return Windows(rolling=roll, lagged=lag)


@dataclass(frozen=True)
class SplitConfig:
    """Схема нарезки истории на обучающие якоря.

    Главное ограничение задачи: истории всего 409 дней. Якорю нужно `max_history`
    дней признаков, а каждый следующий якорь отстоит на `stride` дней назад,
    поэтому глубина окна и число обучающих якорей конкурируют за один ресурс.
    При строгом режиме и окне 365 дней обучающих якорей не остаётся вовсе.
    """

    max_history: int = 180       # глубина окна признаков
    stride: int = 30             # шаг между якорями
    n_train_anchors: int | None = 6   # None — взять максимум помещающихся
    horizon: int = HORIZON
    apply_selection: bool = True      # применять правило отбора на каждом якоре
    # Признаки динамического состояния (EWMA + байесовские фильтры активности
    # и конверсии). Замер на одном якоре: +0.0033 к shape, и они забирают
    # верхние места по важности в обеих головах.
    with_state: bool = True
    # Разрешить якоря с неполной историей. Признаки тогда считаются по тому,
    # что есть; глубину видно модели через hist_span и avail_history.
    allow_partial_history: bool = True
    val_anchor: int = ANCHOR_VAL
    final_anchor: int = ANCHOR_FINAL

    def earliest_anchor(self) -> int:
        """Левая граница якоря. Окно правила отбора нужно всегда."""
        if self.allow_partial_history:
            return SELECTION_SPAN - 1
        return max(self.max_history, SELECTION_SPAN) - 1

    def train_anchors(self) -> list[int]:
        """Обучающие якоря — назад от валидационного с шагом stride."""
        earliest = self.earliest_anchor()
        limit = self.n_train_anchors if self.n_train_anchors is not None else 10**9
        out: list[int] = []
        a = self.val_anchor - self.stride
        while a >= earliest and len(out) < limit:
            out.append(a)
            a -= self.stride
        return sorted(out)

    def max_train_anchors(self) -> int:
        """Сколько якорей вообще помещается при такой глубине окна."""
        return len(SplitConfig(**{**self.__dict__, "n_train_anchors": None}).train_anchors())

    def refit_anchors(self) -> list[int]:
        """Якоря для финального обучения: те же плюс валидационный.

        Перед предсказанием боевого окна валидационный якорь уже не нужен
        как holdout, и выбрасывать самый свежий пример было бы расточительно.
        """
        return sorted([*self.train_anchors(), self.val_anchor])


@dataclass
class ModelConfig:
    """Гиперпараметры двух частей hurdle. Сознательно консервативные."""

    clf_params: dict = field(default_factory=lambda: dict(
        objective="binary", learning_rate=0.05, num_leaves=63,
        min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=5.0, n_estimators=600, verbose=-1, n_jobs=-1,
    ))
    reg_params: dict = field(default_factory=lambda: dict(
        objective="regression", metric="l2", learning_rate=0.05, num_leaves=63,
        min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=5.0, n_estimators=600, verbose=-1, n_jobs=-1,
    ))
    seed: int = 42
    # Ранняя остановка на отложенной доле трейна. Замер на одном якоре:
    # фиксированные 600 деревьев дают shape 1.69291, ранняя остановка
    # (оптимум 149 и 87 деревьев) — 1.68657. Модель была переобучена.
    early_stopping_rounds: int | None = 100
    eval_frac: float = 0.12
    # Для финального сабмита: дообучить на 100 % данных с найденным числом деревьев
    refit_full: bool = False
