"""Плотная календарная решётка: полное состояние пользователя по дням.

Все наши последовательностные модели видели время специфически:
Gap-GRU только дни С СОБЫТИЯМИ, сворачивая паузы в одно число, а
PatchFormer агрегировал в недели и месяцы. Полноценной модели на
исходной дневной решётке не было ни разу.

Критично: день БЕЗ СТРОКИ и строка с search=cat=0 — разные состояния.
По аудиту таких неатрибутированных строк 15.2 %, и событийная модель
представляет их совершенно иначе. Здесь их различает row_present.

Признаки на день (14). Исключены точные дубликаты, установленные
аудитом: has_* тождественны [count>0], to_cart и to_ord точно равны
сумме каналов, gmv равен сумме канальных GMV.

    row_present, search, cat, log1p(searches),
    log1p(search_to_cart), log1p(search_to_ord),
    log1p(cat_to_cart), log1p(cat_to_ord),
    log1p(gmv_search), log1p(gmv_cat),
    sin/cos дня недели, sin/cos годовой фазы
"""
import sys, warnings, gc, time, os; sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import SplitConfig, load_panel, build_anchor

OUT = Path('artifacts/neural/dense'); OUT.mkdir(parents=True, exist_ok=True)
T = 384
COLS = ['searches', 'search_to_cart', 'search_to_ord',
        'cat_to_cart', 'cat_to_ord', 'gmv_search', 'gmv_cat']
NF = 1 + 2 + len(COLS) + 4          # present + флаги + счётчики + календарь


def build(df, A, sp):
    lo = A - T + 1
    if A == sp.final_anchor:
        v = build_anchor(df, A, sp, None, with_target=False)
    else:
        v = build_anchor(df, A, sp, None)
    uid = v.X['user_id'].to_numpy()
    pos = pl.DataFrame({'user_id': uid, '_i': np.arange(len(uid), dtype='uint32')})
    w = (df.filter(pl.col('d').is_between(lo, A))
           .join(pos, on='user_id', how='inner')
           .with_columns(_t=(pl.col('d') - lo).cast(pl.Int32)))
    X = np.zeros((len(uid), T, NF), dtype='float16')
    i = w['_i'].to_numpy(); t = w['_t'].to_numpy()
    X[i, t, 0] = 1.0                                    # строка есть
    X[i, t, 1] = w['search'].to_numpy()
    X[i, t, 2] = w['cat'].to_numpy()
    for k, c in enumerate(COLS):
        X[i, t, 3 + k] = np.log1p(w[c].to_numpy().astype('float32'))
    days = np.arange(lo, A + 1, dtype='float32')
    dow = days % 7
    X[:, :, 3 + len(COLS) + 0] = np.sin(2 * np.pi * dow / 7)
    X[:, :, 3 + len(COLS) + 1] = np.cos(2 * np.pi * dow / 7)
    X[:, :, 3 + len(COLS) + 2] = np.sin(2 * np.pi * days / 365)
    X[:, :, 3 + len(COLS) + 3] = np.cos(2 * np.pi * days / 365)
    np.save(OUT / f'x_a{A}.npy', X)
    np.savez_compressed(OUT / f'meta_a{A}.npz', user_id=uid, lo=lo, hi=A, nf=NF)
    frac = float((X[:, :, 0] > 0).mean())
    return len(uid), frac


if __name__ == '__main__':
    df = load_panel()
    sp = SplitConfig(max_history=300, with_state=True)
    for A in (258, 288, 318, 348, 378, 408):
        t0 = time.perf_counter()
        n, frac = build(df, A, sp)
        sz = (OUT / f'x_a{A}.npy').stat().st_size / 1e9
        print(f'якорь {A}: {n:,} пользователей · дней с активностью {frac:.3f} · '
              f'{sz:.2f} ГБ · {time.perf_counter()-t0:.0f}с', flush=True)
        gc.collect()
    print('готово', flush=True)
