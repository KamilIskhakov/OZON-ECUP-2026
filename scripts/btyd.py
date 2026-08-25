"""Настоящий BG/NBD плюс усадка чека — как отдельная модель.

BTYD у нас использовался лишь как идея для четырёх запросов Gap-GRU,
причём параметры там не постулировались. Полноценной подгонки
BG/NBD никогда не делалось.

На каждом якоре по истории покупок строим тройку (x, t_x, T) и
подгоняем общие (r, alpha, a, b) максимизацией логправдоподобия:

    L = B(a, b+x)/B(a,b) * Г(r+x) alpha^r / (Г(r) (alpha+T)^(r+x))
      + [x>0] B(a+1, b+x-1)/B(a,b) * Г(r+x) alpha^r / (Г(r) (alpha+t_x)^(r+x))

Отсюда P(alive) и E[N_30]. Денежная часть — усадка среднего чека к
общему среднему с весом по числу покупок.

Прогноз BTYD НЕ отдаётся напрямую: проверяется как направление и как
четыре признака поверх СИЛЬНОЙ базы. Шлюз — маржинал 2-3e-4 на обоих
якорях, иначе закрыть.
"""
import sys, warnings, os; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from scipy.special import gammaln, betaln, hyp2f1
from scipy.optimize import minimize
from ecup import load_panel

O = Path('artifacts/neural'); H = 30
df = load_panel()


def rft(A, uid):
    """(x, t_x, T) по покупкам до якоря A включительно."""
    # Событие — ЗАКАЗ, а не покупочный день: у нас есть to_ord внутри дня,
    # и для BG/NBD естественная единица именно транзакция.
    w = (df.filter((pl.col('d') <= A) & (pl.col('to_ord') > 0))
           .group_by('user_id')
           .agg(n=pl.col('to_ord').sum().cast(pl.Float64),
                first=pl.col('d').min(), last=pl.col('d').max()))
    j = pl.DataFrame({'user_id': uid}).join(w, on='user_id', how='left')
    n = j['n'].fill_null(0).to_numpy().astype('float64')
    f = j['first'].fill_null(A).to_numpy().astype('float64')
    l = j['last'].fill_null(A).to_numpy().astype('float64')
    x = np.maximum(n - 1.0, 0.0)      # ПОВТОРНЫЕ покупки
    tx = np.where(n > 0, l - f, 0.0)  # время последней относительно первой
    T = np.where(n > 0, A - f, 0.0)   # возраст с первой покупки
    return x, tx, T


def nll(p, x, tx, T):
    r, al, a, b = np.exp(p)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(al)
    A2 = -(r + x) * np.log(al + T)
    A3 = betaln(a, b + x) - betaln(a, b)
    t1 = A1 + A2 + A3
    with np.errstate(divide='ignore'):
        A4 = betaln(a + 1, b + x - 1) - betaln(a, b)
        t2 = np.where(x > 0, A1 - (r + x) * np.log(al + tx) + A4, -np.inf)
    m = np.maximum(t1, t2)
    ll = m + np.log(np.exp(t1 - m) + np.where(np.isfinite(t2), np.exp(t2 - m), 0.0))
    return -float(ll.sum())


def fit(x, tx, T):
    best, bp = np.inf, None
    for p0 in ([0.0, 2.0, 0.0, 0.0], [-0.5, 3.0, -0.5, 0.5]):
        res = minimize(nll, p0, args=(x, tx, T), method='Nelder-Mead',
                       options=dict(maxiter=4000, fatol=1e-6, xatol=1e-6))
        if res.fun < best:
            best, bp = res.fun, res.x
    return np.exp(bp), best


def predict(par, x, tx, T, t=H):
    r, al, a, b = par
    # P(alive) = 1 / (1 + [x>0] a/(b+x-1) ((alpha+T)/(alpha+t_x))^(r+x)).
    # Отношение (alpha+T)/(alpha+t_x) РАСТЁТ с давностью последней покупки,
    # поэтому в знаменателе стоит exp(+lg). Обратный знак не наказывал
    # уснувших: P(alive) выходила около единицы у всех покупавших.
    lg = (r + x) * (np.log(al + T) - np.log(al + tx))
    denom = 1.0 + np.where(x > 0, (a / np.maximum(b + x - 1, 1e-9)) * np.exp(lg), 0.0)
    p_alive = 1.0 / denom
    f = hyp2f1(r + x, b + x, a + b + x - 1.0, t / (al + T + t))
    # (a-1) законно ОТРИЦАТЕЛЕН при a < 1, что для BG/NBD обычно.
    # Обрезка снизу ломает формулу: числитель тогда взрывается.
    num = ((a + b + x - 1.0) / (a - 1.0)) * (
        1.0 - ((al + T) / (al + T + t)) ** (r + x) * f)
    out = num / denom
    return p_alive, np.nan_to_num(np.clip(out, 0, 100.0))



def main():
  for A in (348, 378):
    s = np.load(O / f'oofpm_strong_a{A}.npz'); uid = s['user_id']
    z = np.log1p(s['y']); base = s['z0'].astype('float64')
    x, tx, T = rft(A, uid)
    # BTYD определена только для совершивших хотя бы одну покупку.
    # Непокупатели (T = 0) вырождают правдоподобие.
    buy = T > 0
    idx = np.flatnonzero(buy)
    sub = np.random.default_rng(0).choice(idx, min(60000, len(idx)), replace=False)
    par, f = fit(x[sub], tx[sub], T[sub])
    pa = np.zeros(len(x)); en = np.zeros(len(x))
    pa[buy], en[buy] = predict(par, x[buy], tx[buy], T[buy])
    print(f'  покупателей в подгонке {buy.mean():.3f}', flush=True)
    # усадка чека: среднее по покупкам пользователя к общему с весом n
    w = (df.filter((pl.col('d') <= A) & (pl.col('gmv') > 0)).group_by('user_id')
           .agg(s=pl.col('gmv').sum(), n=pl.col('d').n_unique()))
    j = pl.DataFrame({'user_id': uid}).join(w, on='user_id', how='left')
    gs = j['s'].fill_null(0.0).to_numpy(); nn = j['n'].fill_null(0).to_numpy()
    glob = gs.sum() / max(nn.sum(), 1)
    K = 3.0
    aov = (gs + K * glob) / (nn + K)
    print(f'\n=== якорь {A} · r {par[0]:.3f} alpha {par[1]:.2f} '
          f'a {par[2]:.3f} b {par[3]:.3f} · -LL {f:,.0f} ===', flush=True)
    print(f'  P(alive): среднее {pa.mean():.3f} · E[N30] среднее {en.mean():.3f} '
          f'(факт покупок за окно неизвестен здесь)', flush=True)
    e = z - base; ec = e - e.mean()
    for nm, v in (('P_alive', pa), ('E[N30]', en), ('log AOV', np.log1p(aov)),
                  ('log GMV30', np.log1p(en * aov))):
        vc = v - v.mean()
        C = float((ec * vc).mean()); V = float((vc ** 2).mean())
        g = C * C / max(V, 1e-15)
        print(f'  {nm:<10} corr с остатком {np.corrcoef(ec, vc)[0,1]:+.4f} · '
              f'одиночный ресурс {e.std()-np.sqrt(max(e.var()-g,0)):+.6f}', flush=True)
    np.savez_compressed(O / f'btyd_a{A}.npz', user_id=uid, p_alive=pa,
                        en30=en, aov=aov, par=par)

if __name__ == '__main__':
    main()
    print('\nсохранены btyd_a{348,378}.npz', flush=True)
