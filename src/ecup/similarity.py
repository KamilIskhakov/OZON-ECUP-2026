"""Похожесть пользователей через траектории поведения.

Классический коллаборативный подход здесь неприменим: item-level данных нет —
ни товаров, ни категорий, ни запросов, только дневные агрегаты. Поэтому
похожесть строится не на «что покупали вместе», а на форме поведения во времени.

Две представленные семьи, и они дают качественно разное:

* **SVD по недельным магнитудам** — раскладывает user × неделя. Первая компонента
  забирает почти всё и означает просто «размер» пользователя (|ρ| с таргетом 0.65,
  как у gmv_365), остальные ~0.03–0.10, то есть шум. Как источник новых признаков
  почти бесполезна, но полезна как контроль.
* **LSA по биграммам дневных токенов** — раскладывает user × переход состояний.
  Даёт многомерную структуру: |ρ| первых компонент 0.50, 0.32, 0.25, 0.21.
  Здесь кодируется не сколько человек тратит, а как он движется по воронке.

Базис (правые сингулярные векторы) обучается один раз на опорном якоре и
переиспользуется на всех остальных. Иначе номер компоненты означал бы на разных
якорях разное, и склейка якорей ломалась бы. Разложение неконтролируемое,
таргет в него не входит, утечки нет.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Дневной токен: биты (search, cat, cart>0, ord>0) плюс 0 = дня нет в данных.
N_TOKENS = 17
N_BIGRAMS = N_TOKENS * N_TOKENS


def _row_index(users: pl.Series, user_col: np.ndarray) -> np.ndarray:
    pos = {u: i for i, u in enumerate(users.to_numpy())}
    return np.fromiter((pos[u] for u in user_col), dtype=np.int64, count=len(user_col))


def token_bigram_matrix(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    span: int = 182,
) -> csr_matrix:
    """Счётчики переходов «состояние сегодня → состояние завтра», N × 289.

    Отсутствие дня — полноправный токен: паузы несут не меньше информации,
    чем визиты, и без них последовательность теряет ритм.
    """
    n = len(users)
    t = (
        df.filter(pl.col("user_id").is_in(users) & pl.col("d").is_between(anchor - span + 1, anchor))
          .select(["user_id", "d", "search", "cat", "to_cart", "to_ord"])
          .sort(["user_id", "d"])
    )
    code = (
        t["search"].to_numpy().astype("int16")
        + t["cat"].to_numpy().astype("int16") * 2
        + (t["to_cart"].to_numpy() > 0).astype("int16") * 4
        + (t["to_ord"].to_numpy() > 0).astype("int16") * 8
        + 1
    )
    seq = np.zeros((n, span), dtype="int8")
    seq[_row_index(users, t["user_id"].to_numpy()), t["d"].to_numpy() - (anchor - span + 1)] = code

    cur, nxt = seq[:, :-1].ravel().astype("int32"), seq[:, 1:].ravel().astype("int32")
    rows = np.repeat(np.arange(n), span - 1)
    m = csr_matrix((np.ones(len(cur), dtype="float32"), (rows, cur * N_TOKENS + nxt)),
                   shape=(n, N_BIGRAMS))
    m.sum_duplicates()
    return m


def weekly_matrix(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    n_weeks: int = 26,
    cols: tuple[str, ...] = ("gmv", "to_ord", "to_cart", "searches"),
) -> csr_matrix:
    """Недельные магнитуды в log-шкале, N × (n_weeks · len(cols)).

    Неделя 0 — самая свежая, отсчёт назад от якоря, поэтому колонки
    сопоставимы между якорями.
    """
    n = len(users)
    h = (
        df.filter(pl.col("user_id").is_in(users)
                  & pl.col("d").is_between(anchor - n_weeks * 7 + 1, anchor))
          .with_columns(w=((anchor - pl.col("d")) // 7).cast(pl.Int16))
    )
    agg = h.group_by(["user_id", "w"]).agg([pl.col(c).sum() for c in cols])
    ri = _row_index(users, agg["user_id"].to_numpy())
    wi = agg["w"].to_numpy().astype(np.int64)

    blocks = []
    for k, c in enumerate(cols):
        v = np.log1p(agg[c].to_numpy().astype("float64"))
        blocks.append(csr_matrix((v, (ri, wi + k * n_weeks)), shape=(n, n_weeks * len(cols))))
    m = sum(blocks).tocsr()
    m.sum_duplicates()
    return m


@dataclass
class LSABasis:
    """Фиксированный базис разложения: компоненты, idf и имена признаков."""

    components: np.ndarray        # (k, n_features) — правые сингулярные векторы
    idf: np.ndarray | None
    prefix: str

    @property
    def names(self) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(self.components.shape[0])]

    def transform(self, m: csr_matrix) -> np.ndarray:
        if self.idf is not None:
            m = m.multiply(self.idf).tocsr()
        return np.asarray(m @ self.components.T, dtype="float32")


def fit_lsa(m: csr_matrix, k: int = 16, prefix: str = "lsa", use_idf: bool = True) -> LSABasis:
    """Усечённое SVD с опциональным idf-взвешиванием.

    Знаки и порядок компонент у svds произвольны, поэтому фиксируем:
    сортировка по убыванию сингулярных чисел и знак по большей массе.
    Без этого номер компоненты означал бы на разных запусках разное.
    """
    idf = None
    if use_idf:
        n = m.shape[0]
        idf = np.log(n / (1.0 + np.asarray((m > 0).sum(0)).ravel())).astype("float64")
        m = m.multiply(idf).tocsr()
    k = min(k, min(m.shape) - 1)
    _, s, vt = svds(m, k=k)
    order = np.argsort(-s)
    vt = vt[order]
    flip = np.where(vt.sum(axis=1) < 0, -1.0, 1.0)[:, None]
    return LSABasis(components=(vt * flip).astype("float64"), idf=idf, prefix=prefix)


def build_embeddings(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    token_basis: LSABasis | None = None,
    weekly_basis: LSABasis | None = None,
    token_span: int = 182,
    n_weeks: int = 26,
) -> pl.DataFrame:
    """Эмбеддинги популяции якоря по заранее обученным базисам."""
    out = {"user_id": users}
    if token_basis is not None:
        e = token_basis.transform(token_bigram_matrix(df, anchor, users, token_span))
        out |= {n: e[:, i] for i, n in enumerate(token_basis.names)}
    if weekly_basis is not None:
        e = weekly_basis.transform(weekly_matrix(df, anchor, users, n_weeks))
        out |= {n: e[:, i] for i, n in enumerate(weekly_basis.names)}
    return pl.DataFrame(out).sort("user_id")


def fit_bases(
    df: pl.DataFrame,
    anchor: int,
    users: pl.Series,
    k_token: int = 16,
    k_weekly: int = 8,
    token_span: int = 182,
    n_weeks: int = 26,
) -> tuple[LSABasis, LSABasis]:
    """Обучить оба базиса на опорном якоре. Таргет не используется."""
    tb = fit_lsa(token_bigram_matrix(df, anchor, users, token_span), k=k_token,
                 prefix="tok", use_idf=True)
    wb = fit_lsa(weekly_matrix(df, anchor, users, n_weeks), k=k_weekly,
                 prefix="wk", use_idf=False)
    return tb, wb


def neighbor_target_features(
    emb_ref: np.ndarray,
    y_ref: np.ndarray,
    emb_query: np.ndarray,
    k: int = 50,
) -> pl.DataFrame:
    """Исходы похожих пользователей как признаки.

    Соседи берутся ТОЛЬКО из другого якоря, у которого таргет уже наблюдён.
    Искать соседей внутри своего якоря нельзя: их таргет — часть того же
    окна, которое мы предсказываем, и это прямая утечка.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(emb_ref)
    dist, idx = nn.kneighbors(emb_query)
    z = np.log1p(y_ref)[idx]
    return pl.DataFrame({
        "nn_mean_z": z.mean(1),
        "nn_median_z": np.median(z, axis=1),
        "nn_std_z": z.std(1),
        "nn_zero_rate": (z == 0).mean(1),
        "nn_dist_mean": dist.mean(1),
    })
