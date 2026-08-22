"""Низкочастотная форма всей блочной истории: линейный контроль на DCT.

Новый источник X: не последние 300 дней и не выбранная вручную гармоника,
а полная последовательность 30-дневных блоков с ЯВНО отделённым уровнем
пользователя. v22 показал, что смешивание уровня и формы маскирует сигнал.

Ridge, а не GBDT: в неудачном longhist жадное дерево выбрало уровневые
агрегаты и проигнорировало слабую форму.

Общая опора: k = 1..9, то есть дни A-270..A-30. Для якоря 318 это
минимальная левая граница, глубже блоки уходят в отрицательные дни.
"""
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from scipy.fft import dct
from ecup import load_panel
from ecup.directions import marginal_gain
df = load_panel()
K, NC = 9, 5                       # блоков и оставляемых DCT-компонент
QTY = {'g': pl.col('gmv').sum(), 'n': pl.col('to_ord').sum(),
       'b': (pl.col('gmv')>0).sum(), 's': pl.col('searches').sum()}

def prep(A):
    o = np.load(f'artifacts/neural/oof_a{A}.npz'); uid = o['user_id']
    e = np.log1p(o['y']) - o['z0']; u = pl.Series('user_id', uid)
    seq = {q: [] for q in QTY}
    for k in range(1, K+1):
        lo, hi = A-30*(k+1)+1, A-30*k
        assert lo >= 0, (A, k, lo)
        t=(df.filter(pl.col('d').is_between(lo,hi)&pl.col('user_id').is_in(u))
             .group_by('user_id').agg(**QTY))
        r=(pl.DataFrame({'user_id':uid}).join(t,on='user_id',how='left')
             .with_columns(pl.exclude('user_id').fill_null(0)).sort('user_id'))
        for q in QTY: seq[q].append(np.log1p(r[q].to_numpy().astype('float64')))
    F, N = [], []
    for q in QTY:
        B = np.column_stack(seq[q][::-1])          # от старого к новому
        lvl = B.mean(1)
        Bc = B - lvl[:, None]                      # уровень отделён явно
        D = dct(Bc, axis=1, norm='ortho')[:, 1:NC+1]
        F.append(lvl); N.append(f'{q}_level')
        for c in range(NC): F.append(D[:, c]); N.append(f'{q}_dct{c+1}')
    return e, np.column_stack(F).astype('float64'), N

FOLDS = [([318], 348), ([318, 348], 378)]
d16 = np.load('artifacts/neural/dz_a378.npz')['dz']
dann = np.load('/tmp/d_annual_378.npy')
LAM = 30.0
print(f'блоков {K} (дни A-270…A-30) · компонент DCT {NC} · признаков {4*(NC+1)}\n')
print(f'{"фолд":<20} {"alpha":>9} {"сольно":>10} {"маржин":>10}  базис')
for tr, te in FOLDS:
    Xs, ys = [], []
    for A in tr:
        e, X, N = prep(A); Xs.append(X); ys.append(e - e.mean())
    X = np.vstack(Xs); y = np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0)+1e-9; Xz = (X-mu)/sd
    coef = np.linalg.solve(Xz.T@Xz + LAM*np.eye(Xz.shape[1]), Xz.T@y)
    e_t, Xt, _ = prep(te)
    d = ((Xt-mu)/sd) @ coef
    ex = [d16, dann] if te == 378 else []
    r = marginal_gain(e_t, d, existing=ex)
    print(f'  {str(tr)+" -> "+str(te):<18} {r["alpha_signed"]:>+9.4f} '
          f'{r["gain_solo"]:>+10.5f} {r["gain_marginal"]:>+10.5f}  '
          f'{"нейро+годовой" if ex else "пусто"}')
    if te == 378:
        top = sorted(zip(N, np.abs(coef)), key=lambda x: -x[1])[:6]
        print(f'    вес: ' + ', '.join(f'{n}={v:.3f}' for n, v in top))
