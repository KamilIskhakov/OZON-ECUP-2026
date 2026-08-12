"""ecup — решение задачи OZON E-CUP 2026 «Search LTV».

Прогноз суммарного GMV пользователя за 30 дней, метрика RMSLE.

Быстрый старт:

    from ecup import load_panel, run_validation, SplitConfig

    df = load_panel()
    res = run_validation(df=df, split=SplitConfig(max_history=365, n_train_anchors=6))
"""
from .config import (ANCHOR_FINAL, ANCHOR_VAL, HORIZON, ModelConfig,
                     SplitConfig, Windows, anchor_label, d_to_date, date_to_d,
                     target_window)
from .data import (all_users, anchor_population, load_panel, make_target,
                   selected_users)
from .dataset import (AnchorFrame, anchor_weights, build_anchor,
                      build_training_set, to_matrix)
from .features import build_features, feature_names
from .metrics import (best_constant, correct_oversampling_prior, hurdle_glue,
                      hurdle_glue_naive, level_shift_cost, rmsle, summarize)
from .model import DirectGBDT, HurdleGBDT
from .pipeline import (ValidationResult, make_submission, run_validation,
                       sweep_anchors, sweep_history)

__all__ = [
    "ANCHOR_FINAL", "ANCHOR_VAL", "HORIZON", "ModelConfig", "SplitConfig", "Windows",
    "anchor_label", "d_to_date", "date_to_d", "target_window",
    "all_users", "anchor_population", "load_panel", "make_target", "selected_users",
    "AnchorFrame", "anchor_weights", "build_anchor", "build_training_set", "to_matrix",
    "build_features", "feature_names",
    "best_constant", "correct_oversampling_prior", "hurdle_glue", "hurdle_glue_naive",
    "level_shift_cost", "rmsle", "summarize",
    "DirectGBDT", "HurdleGBDT",
    "ValidationResult", "make_submission", "run_validation", "sweep_anchors", "sweep_history",
]
__version__ = "0.1.0"
