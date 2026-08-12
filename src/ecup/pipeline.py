"""Сценарии: валидация, эксперимент по длине окна, финальный сабмит."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .config import (ARTIFACTS, ModelConfig, SAMPLE_SUBMIT, SplitConfig,
                     Windows, anchor_label, target_window)
from .data import anchor_population, load_panel, make_target
from .dataset import (anchor_offsets, anchor_weights, build_anchor,
                      build_training_set, to_matrix)
from .metrics import (best_constant, hurdle_glue, hurdle_glue_naive, rmsle,
                      summarize)
from .model import DirectGBDT, HurdleGBDT


def bias_shape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """(bias, shape_rmse) — разложение ошибки на уровень и форму.

    e = z − ẑ; bias = mean(e) чинится одним числом, shape = std(e) чинится
    только моделью. Выбирать конфигурации надо по shape, а календарный
    уровень решать отдельно: иначе хорошая модель с плохим уровнем
    проигрывает плохой модели со случайно удачным уровнем.
    """
    e = (np.log1p(np.asarray(y_true, "float64"))
         - np.log1p(np.clip(np.asarray(y_pred, "float64"), 0, None)))
    return float(e.mean()), float(e.std())


def delevel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE после снятия постоянного смещения в log-шкале.

    Разделяет две разные ошибки: насколько модель верно ранжирует пользователей
    и насколько верно угадала общий уровень целевого окна. Уровень чинится одним
    числом, ранжирование — только моделью. Оценка оптимистична (сдвиг подобран
    по факту), но именно она показывает потолок калибровки уровня.
    """
    zt = np.log1p(np.asarray(y_true, "float64"))
    zp = np.log1p(np.clip(np.asarray(y_pred, "float64"), 0, None))
    return float(np.sqrt(np.mean((zp - zt - (zp - zt).mean()) ** 2)))


@dataclass
class ValidationResult:
    split: SplitConfig
    rmsle_hurdle: float
    rmsle_direct: float
    rmsle_naive_glue: float
    rmsle_constant: float
    rmsle_hurdle_delevel: float
    n_train: int
    n_val: int
    n_features: int
    diagnostics: dict = field(default_factory=dict)
    model: HurdleGBDT | None = None
    val_frame: pl.DataFrame | None = None
    seconds: float = 0.0

    def as_row(self) -> dict:
        return {
            "max_history": self.split.max_history,
            "n_anchors": len(self.split.train_anchors()),
            "full_hist": not self.split.allow_partial_history,
            "n_train": self.n_train,
            "n_features": self.n_features,
            "hurdle": round(self.rmsle_hurdle, 5),
            "direct": round(self.rmsle_direct, 5),
            "hurdle_delevel": round(self.rmsle_hurdle_delevel, 5),
            "naive_glue": round(self.rmsle_naive_glue, 5),
            "constant": round(self.rmsle_constant, 5),
            "sec": round(self.seconds, 1),
        }


def run_validation(
    df: pl.DataFrame | None = None,
    split: SplitConfig | None = None,
    model_config: ModelConfig | None = None,
    windows: Windows | None = None,
    use_weights: bool = True,
    normalize_level: bool = True,
    drop_anchor_calendar: bool = True,
    keep_model: bool = False,
    verbose: bool = True,
) -> ValidationResult:
    """Обучить на исторических якорях, померить на валидационном.

    Валидация строго по времени: таргет-окно валидационного якоря лежит правее
    всех обучающих таргет-окон, пересечений нет.

    `normalize_level` вычитает из цели средний уровень якоря, приводя окна
    с разным сезонным спросом к общей шкале. Уровень целевого окна возвращается
    при предсказании; по умолчанию берётся уровень самого свежего якоря.
    """
    t0 = time.perf_counter()
    df = load_panel() if df is None else df
    split = split or SplitConfig()
    model_config = model_config or ModelConfig()

    anchors = split.train_anchors()
    if not anchors:
        raise ValueError(f"при max_history={split.max_history} не осталось обучающих якорей")
    if verbose:
        print(f"обучающие якоря ({len(anchors)}): "
              f"{', '.join(anchor_label(a) for a in anchors)}")

    Xtr_df, ytr, aid, levels = build_training_set(df, anchors, split, windows, verbose=verbose)
    Xtr, feats = to_matrix(Xtr_df, drop_anchor_calendar=drop_anchor_calendar)
    w = anchor_weights(aid) if use_weights else None

    # Каждая голова нормируется на свою маржу: классификатор на p̄, регрессия
    # на ℓ⁺. Целевые значения берём с самого свежего якоря — он ближайший
    # по сезону и по составу к тому, что предстоит предсказывать.
    last = levels[max(anchors)]
    if normalize_level:
        clf_init, z_off = anchor_offsets(aid, levels)
        p_target, m_offset = last.p_bar, last.l_plus
    else:
        clf_init, z_off, p_target, m_offset = None, None, None, 0.0

    val = build_anchor(df, split.val_anchor, split, windows)
    Xva, _ = to_matrix(val.X, feats)
    yva = val.y

    hur = HurdleGBDT(config=model_config).fit(
        Xtr, ytr, feature_names=feats, sample_weight=w,
        z_offset=z_off, clf_init=clf_init)
    dir_ = DirectGBDT(config=model_config).fit(
        Xtr, ytr, feature_names=feats, sample_weight=w,
        z_offset=None if z_off is None else np.full(len(ytr), last.l))

    p, m = hur.predict_parts(Xva, p_target=p_target, m_offset=m_offset)
    m = np.clip(m, 0.0, None)
    pred_hurdle = hurdle_glue(p, m)
    pred_naive = hurdle_glue_naive(p, m)
    pred_direct = dir_.predict(Xva, level_shift=0.0 if z_off is None else last.l)

    res = ValidationResult(
        split=split,
        rmsle_hurdle=rmsle(yva, pred_hurdle),
        rmsle_direct=rmsle(yva, pred_direct),
        rmsle_naive_glue=rmsle(yva, pred_naive),
        rmsle_constant=rmsle(yva, np.full_like(yva, best_constant(ytr))),
        rmsle_hurdle_delevel=delevel(yva, pred_hurdle),
        n_train=len(ytr), n_val=len(yva), n_features=len(feats),
        diagnostics={"hurdle": summarize(yva, pred_hurdle),
                     "direct": summarize(yva, pred_direct),
                     "p_mean": float(p.mean()), "m_mean": float(m.mean()),
                     "p_target": p_target, "m_offset": m_offset,
                     "p_bar_true_val": float((yva > 0).mean()),
                     "bias": bias_shape(yva, pred_hurdle)[0],
                     "shape": bias_shape(yva, pred_hurdle)[1]},
        model=hur if keep_model else None,
        val_frame=val.X.select("user_id").with_columns(
            y=pl.Series(yva), p=pl.Series(p), m=pl.Series(m),
            pred_hurdle=pl.Series(pred_hurdle),
            pred_direct=pl.Series(pred_direct)) if keep_model else None,
        seconds=time.perf_counter() - t0,
    )
    if verbose:
        print(f"  hurdle {res.rmsle_hurdle:.5f} | без смещения уровня "
              f"{res.rmsle_hurdle_delevel:.5f} | direct {res.rmsle_direct:.5f} | "
              f"наивная склейка {res.rmsle_naive_glue:.5f} | "
              f"константа {res.rmsle_constant:.5f} | {res.seconds:.0f}с")
    return res


def sweep_history(
    history_grid: tuple[int, ...] = (60, 90, 120, 168, 240, 300, 365),
    n_train_anchors: int | None = 4,
    allow_partial_history: bool = True,
    df: pl.DataFrame | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Проверить, какая глубина окна признаков реально нужна.

    Два режима, и они отвечают на разные вопросы.

    `n_train_anchors` задан числом — сравниваются именно окна: обучающая выборка
    одинакова по объёму, меняется только глубина признаков.

    `n_train_anchors=None` — сравнивается практический компромисс: длинное окно
    съедает историю, и обучающих якорей помещается меньше. Именно из-за этого
    ограничения победитель GA Customer Revenue взял 168 дней — у него была
    своя длина истории, и переносить это число к нам без проверки нельзя.
    """
    df = load_panel() if df is None else df
    rows = []
    for h in history_grid:
        split = SplitConfig(max_history=h, n_train_anchors=n_train_anchors,
                            allow_partial_history=allow_partial_history)
        n = len(split.train_anchors())
        if n == 0:
            if verbose:
                print(f"\n=== окно {h} дней: обучающих якорей не остаётся, пропуск ===")
            continue
        if verbose:
            print(f"\n=== окно признаков {h} дней · якорей {n} ===")
        rows.append(run_validation(df=df, split=split, verbose=verbose).as_row())
    return pl.DataFrame(rows).sort("hurdle")


def sweep_anchors(
    anchor_grid: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    max_history: int = 365,
    df: pl.DataFrame | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Сколько исторических якорей окупается."""
    df = load_panel() if df is None else df
    rows = []
    for n in anchor_grid:
        if verbose:
            print(f"\n=== {n} обучающих якорей ===")
        split = SplitConfig(max_history=max_history, n_train_anchors=n)
        try:
            rows.append(run_validation(df=df, split=split, verbose=verbose).as_row())
        except ValueError as e:
            print(f"  пропуск: {e}")
    return pl.DataFrame(rows).sort("hurdle")


def make_submission(
    split: SplitConfig | None = None,
    model_config: ModelConfig | None = None,
    windows: Windows | None = None,
    a_p: float = 0.0,
    a_m: float = 0.0,
    blend_alpha: float = 0.0,
    normalize_level: bool = True,
    out_name: str = "submission.csv",
    df: pl.DataFrame | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """Переобучить на всех якорях, включая валидационный, и предсказать боевое окно.

    Уровень целевого окна задаётся ДВУМЯ ручками, по одной на маржу:

    * `a_p` — сдвиг логита базовой доли покупателей (экстенсивная маржа);
    * `a_m` — сдвиг условного среднего ℓ⁺ (интенсивная маржа).

    Разложение прошлогоднего перехода на СОПОСТАВИМОЙ популяции (с фильтром
    активности) даёт a_p ≈ 0 и a_m ≈ +0.05: у уже активных пользователей
    вероятность покупки между январём и февралём-мартом почти не меняется,
    растёт только сумма. На нефильтрованной популяции картина обратная
    (85 % экстенсивной), но это артефакт смены состава, а наша тестовая
    популяция условна на активности по построению.

    `blend_alpha` — доля direct-модели в смеси, смешивание в z-пространстве.
    """
    df = load_panel() if df is None else df
    split = split or SplitConfig()
    model_config = model_config or ModelConfig()

    anchors = split.refit_anchors()
    if verbose:
        s, e = target_window(split.final_anchor, split.horizon)
        print(f"переобучение на {len(anchors)} якорях, прогноз на {s} … {e}")

    Xtr_df, ytr, aid, levels = build_training_set(df, anchors, split, windows, verbose=verbose)
    Xtr, feats = to_matrix(Xtr_df)

    last = levels[max(anchors)]
    w = anchor_weights(aid)
    if normalize_level:
        clf_init, z_off = anchor_offsets(aid, levels)
        p_base = 1.0 / (1.0 + np.exp(-(np.log(last.p_bar / (1 - last.p_bar)) + a_p)))
        m_off = last.l_plus + a_m
    else:
        clf_init, z_off, p_base, m_off = None, None, None, a_m

    model = HurdleGBDT(config=model_config).fit(
        Xtr, ytr, feature_names=feats, sample_weight=w,
        z_offset=z_off, clf_init=clf_init)
    z_h = np.log1p(model.predict(
        (Xfi := to_matrix(
            (final := build_anchor(df, split.final_anchor, split, windows,
                                   with_target=False)).X, feats)[0]),
        p_target=p_base, m_offset=m_off))

    if blend_alpha > 0:
        direct = DirectGBDT(config=model_config).fit(
            Xtr, ytr, feature_names=feats, sample_weight=w,
            z_offset=None if z_off is None else np.full(len(ytr), last.l))
        z_d = np.log1p(direct.predict(Xfi, level_shift=0.0 if z_off is None else last.l + a_m))
        z = blend_alpha * z_d + (1 - blend_alpha) * z_h
    else:
        z = z_h
    pred = np.expm1(np.clip(z, 0.0, None))

    if verbose:
        print(f"уровень: p̄ {last.p_bar:.4f} + a_p {a_p:+.4f} → {p_base:.4f} · "
              f"ℓ⁺ {last.l_plus:.4f} + a_m {a_m:+.4f} = {m_off:.4f} · "
              f"бленд direct {blend_alpha:.2f}")

    sub = final.X.select("user_id").with_columns(predict=pl.Series(pred))

    # Состав и порядок строк должны в точности повторять sample_submit.csv
    sample = pl.read_csv(SAMPLE_SUBMIT)
    sub = (sample.select("user_id")
                 .join(sub, on="user_id", how="left")
                 .with_columns(pl.col("predict").fill_null(0.0).clip(0.0, None)))
    missing = int(sub["predict"].is_null().sum())
    if missing:
        raise RuntimeError(f"{missing} пользователей без прогноза")

    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / out_name
    sub.write_csv(path)
    if verbose:
        z = np.log1p(sub["predict"].to_numpy())
        print(f"записано {path}  ({sub.height:,} строк)")
        print(f"  доля ненулевых {100*(sub['predict']>0.5).mean():.1f}% · "
              f"mean log1p {z.mean():.4f}")
        ss = np.log1p(sample["predict"].to_numpy())
        print(f"  для сравнения sample_submit: доля ненулевых "
              f"{100*(sample['predict']>0.5).mean():.1f}% · mean log1p {ss.mean():.4f}")
    return sub
