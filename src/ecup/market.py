"""Рыночно-нормированные и беcкэповые долгие признаки.

Две дыры, найденные аудитом 183 базовых признаков.

ПЕРВАЯ: глобального уровня дня в X нет вовсе. Дневной рыночный GMV
растёт с 374 тыс. до 750 тыс. за панель (max/median = 1.65), поэтому
GMV_30 = 100 на раннем и позднем якоре означают РАЗНЫЙ относительный
размер пользователя, а дерево видит одно и то же число. Нормировка
якоря в пайплайне сдвигает ТАРГЕТ одним скаляром, но признаки
остаются сырыми.

Сырые рыночные величины подавать нельзя: внутри якоря они постоянны и
работают как его идентификатор, а удаление anchor_doy в своё время
дало +0.0073. Поэтому рынок входит ТОЛЬКО вычитанием внутри
пользовательского признака.

ВТОРАЯ: max_history обрезает «пожизненные» агрегаты. На якоре 378 при
h=300 медиана hist_span равна 296 и gmv_hist в среднем 895.8, а без
обрезки — 373 и 1067.7. То есть боевая модель с h=300 не видит первые
78 дней, на 408 — первые 108. Ровно ту область, где годовой фильтр
нашёл сигнал.
"""
from __future__ import annotations
import numpy as np
import polars as pl

WIN_REL = (7, 30, 90, 180)
WIN_CNT = (30, 90)


def _market(df: pl.DataFrame) -> pl.DataFrame:
    """Дневной агрегат рынка: GMV, покупатели, активные пользователи."""
    return (df.group_by('d')
              .agg(m_gmv=pl.col('gmv').sum().cast(pl.Float64),
                   m_buy=(pl.col('gmv') > 0).sum().cast(pl.Float64),
                   m_usr=pl.col('user_id').n_unique().cast(pl.Float64))
              .sort('d'))


def market_features(df: pl.DataFrame, anchor: int, users: np.ndarray,
                    mkt: pl.DataFrame | None = None) -> tuple[np.ndarray, list[str]]:
    """Признаки для одного якоря, выровненные по порядку `users`."""
    mkt = _market(df) if mkt is None else mkt
    lo_all = 1
    hist = df.filter(pl.col('d').is_between(lo_all, anchor))
    base = pl.DataFrame({'user_id': users})
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, v: np.ndarray) -> None:
        cols.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)); names.append(name)

    rel: dict[int, np.ndarray] = {}
    for W in WIN_REL:
        lo = anchor - W + 1
        u = (hist.filter(pl.col('d') >= lo).group_by('user_id')
                 .agg(g=pl.col('gmv').sum().cast(pl.Float64)))
        g = base.join(u, on='user_id', how='left')['g'].fill_null(0.0).to_numpy()
        m = mkt.filter(pl.col('d').is_between(lo, anchor))
        # средний пользователь окна: рыночный GMV на одного активного в день
        per = float(m['m_gmv'].sum()) / max(float(m['m_usr'].mean()), 1.0)
        r = np.log1p(g) - np.log1p(per)
        rel[W] = r; add(f'rel_gmv_{W}', r)
    add('rel_acc_30_180', rel[30] - rel[180])
    add('rel_acc_7_90', rel[7] - rel[90])

    for W in WIN_CNT:
        lo = anchor - W + 1
        u = (hist.filter(pl.col('d') >= lo).group_by('user_id')
                 .agg(o=pl.col('to_ord').sum().cast(pl.Float64),
                      b=(pl.col('gmv') > 0).sum().cast(pl.Float64)))
        j = base.join(u, on='user_id', how='left')
        m = mkt.filter(pl.col('d').is_between(lo, anchor))
        po = float(m['m_buy'].sum()) / max(float(m['m_usr'].mean()), 1.0)
        add(f'rel_ord_{W}', np.log1p(j['o'].fill_null(0.0).to_numpy()) - np.log1p(po))
        add(f'rel_buy_{W}', np.log1p(j['b'].fill_null(0.0).to_numpy()) - np.log1p(po))

    # --- беcкэповая история: ВСЕ дни 1..anchor, без ограничения max_history
    half = lo_all + (anchor - lo_all) // 2
    agg = (hist.group_by('user_id')
               .agg(fg=pl.col('gmv').sum().cast(pl.Float64),
                    fb=(pl.col('gmv') > 0).sum().cast(pl.Float64),
                    fo=pl.col('to_ord').sum().cast(pl.Float64),
                    fa=(pl.col('searches') > 0).sum().cast(pl.Float64),
                    eg=(pl.col('gmv') * (pl.col('d') <= half)).sum().cast(pl.Float64),
                    fd=pl.col('d').filter(pl.col('gmv') > 0).min().cast(pl.Float64)))
    j = base.join(agg, on='user_id', how='left')
    span = float(anchor - lo_all + 1)
    fg = j['fg'].fill_null(0.0).to_numpy(); fb = j['fb'].fill_null(0.0).to_numpy()
    fo = j['fo'].fill_null(0.0).to_numpy(); fa = j['fa'].fill_null(0.0).to_numpy()
    eg = j['eg'].fill_null(0.0).to_numpy(); fd = j['fd'].to_numpy().astype('float64')
    add('life_gmv_rate', np.log1p(fg / span))
    add('life_aov', np.log1p(fg / np.maximum(fb, 1.0)))
    add('life_buy_rate', fb / span)
    add('life_ord_rate', fo / span)
    add('life_act_frac', fa / span)
    add('life_early_share', eg / np.maximum(fg, 1e-9))
    add('life_tenure', np.where(np.isfinite(fd), (anchor - fd) / span, 0.0))
    # недавний темп против пожизненного — отношение, а не два уровня
    add('life_rel_recent', rel[90] - np.log1p(fg / span))
    return np.column_stack(cols).astype('float32'), names


# --- чистая версия: систематическая нормировка накопительных величин ---
#
# Дефекты первой версии, найденные при разборе:
#   life_aov       считался как GMV/buydays, а не GMV/orders;
#   life_act_frac  мерил долю дней с ПОИСКОМ, а не активность пайплайна;
#   life_rel_recent непрозрачная смесь единиц;
#   rel_*          рыночный блок дал ноль на обоих якорях.
#
# Определения приведены к пайплайну: активный день — день присутствия
# в панели (в features.py это pl.len()), покупочный — день с gmv > 0.
#
# Два охвата. `full` — все дни 0..T, то есть БЕЗ обрезки max_history.
# `capped` — последние 300 дней, то есть ровно то, что видит боевая
# модель. Сравнение двух охватов разделяет два механизма: если capped
# даёт почти столько же, дело было в единицах измерения; если прирост
# только у full — в старой истории есть новый сигнал.

CAP = 300


def rate_features(df: pl.DataFrame, anchor: int, users: np.ndarray,
                  scopes: tuple[str, ...] = ('full', 'capped')
                  ) -> tuple[np.ndarray, list[str]]:
    base = pl.DataFrame({'user_id': users})
    cols: list[np.ndarray] = []
    names: list[str] = []
    for sc in scopes:
        lo = 0 if sc == 'full' else max(0, anchor - CAP + 1)
        days = float(anchor - lo + 1)
        a = (df.filter(pl.col('d').is_between(lo, anchor)).group_by('user_id')
               .agg(g=pl.col('gmv').sum().cast(pl.Float64),
                    o=pl.col('to_ord').sum().cast(pl.Float64),
                    c=pl.col('to_cart').sum().cast(pl.Float64),
                    s=pl.col('searches').sum().cast(pl.Float64),
                    bd=(pl.col('gmv') > 0).sum().cast(pl.Float64),
                    ad=pl.len().cast(pl.Float64),
                    fd=pl.col('d').min().cast(pl.Float64)))
        j = base.join(a, on='user_id', how='left')
        g = j['g'].fill_null(0.0).to_numpy(); o = j['o'].fill_null(0.0).to_numpy()
        c = j['c'].fill_null(0.0).to_numpy(); s = j['s'].fill_null(0.0).to_numpy()
        bd = j['bd'].fill_null(0.0).to_numpy(); ad = j['ad'].fill_null(0.0).to_numpy()
        fd = j['fd'].to_numpy().astype('float64')
        ten = np.where(np.isfinite(fd), anchor - fd, 0.0)
        for nm_, v in (
            ('gmv_per_day', np.log1p(g / days)),
            ('ord_per_day', np.log1p(o / days)),
            ('cart_per_day', np.log1p(c / days)),
            ('srch_per_day', np.log1p(s / days)),
            ('buyday_frac', bd / days),
            ('actday_frac', ad / days),
            ('gmv_per_buyday', np.log1p(g / np.maximum(bd, 1.0))),
            ('aov', np.log1p(g / np.maximum(o, 1.0))),
            ('ord_per_buyday', o / np.maximum(bd, 1.0)),
            ('tenure_frac', ten / days),
        ):
            cols.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
            names.append(f'{sc[0]}_{nm_}')
    return np.column_stack(cols).astype('float32'), names


def gmv_rate_only(df: pl.DataFrame, anchor: int, users: np.ndarray,
                  scope: str) -> np.ndarray:
    """Один признак GMV/день для теста механизма capped против full."""
    v, _ = rate_features(df, anchor, users, scopes=(scope,))
    return v[:, :1]
