"""Честный сильный z0 на каждом якоре — база для retargeting Gap-GRU.

Сегодня выяснилось, что сеть, обученная на остатке УСТАРЕВШЕЙ базы,
переоткрывает уже занятую ось: corr(d_C, d_GRU1) = 0.58, маржинально
+0.00002 при рекордных +0.00111 standalone. Значит чинить надо цель,
а не архитектуру.

    z0_strong(A) = z0_oof(A) + 0.35 d_GRU1(A) + 0.0104 d_ann(A)
                             + 0.24 d_life(A)

Все три компоненты должны быть ЧЕСТНЫМИ относительно A, иначе сеть
будет учиться на остатке базы, видевшей этот таргет.

  z0_oof   — уже есть в oofpm (пара LGB+CB при 50/50, OOF по построению)
  d_ann    — детерминированный согласованный фильтр, веса гармоники от
             якоря не зависят, блоки сдвигаются вместе с ним
  d_GRU1   — из СОХРАНЁННЫХ чекпоинтов: fold0 обучался на 198..288 и
             потому честен для 318 и 348, fold1 на 198..318 — для 378
  d_life   — считается отдельно (нужен LGB+CB с блоком и без)

Коэффициенты НЕ переоптимизируются на каждом якоре: берутся боевые,
иначе сеть будет исправлять искусственную нестабильность весов.
"""
import sys, warnings; sys.path.insert(0, 'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl
from pathlib import Path
from ecup import load_panel

O = Path('artifacts/neural'); TOK = O / 'tokens'
# Одиннадцать блоков и фаза -364.5 восстановлены сверкой с сохранённым
# d_annual_378.npy: corr 0.99995, std 4.2819 против 4.2921. Вариант из
# regimes.py (девять блоков, фаза -379.5) давал лишь corr 0.74.
JANN = np.arange(0, 11)
WANN = np.cos(2 * np.pi * (-364.5 + 30 * JANN.astype(float)) / 365)
WANN = WANN - WANN.mean()
CK = {318: 'gapgru_evt_ckpt_fold0.pt', 348: 'gapgru_evt_ckpt_fold0.pt',
      378: 'gapgru_evt_ckpt_fold1.pt'}


def annual(df, A, uid):
    """Годовой согласованный фильтр на якоре A, порядок как в uid."""
    u = pl.Series('user_id', uid); cols = []
    for j in JANN:
        t = (df.filter(pl.col('d').is_between(A - 364 + 30 * j, A - 335 + 30 * j)
                       & pl.col('user_id').is_in(u))
               .group_by('user_id').agg(g=pl.col('gmv').sum()))
        r = (pl.DataFrame({'user_id': uid}).join(t, on='user_id', how='left')
               .with_columns(pl.col('g').fill_null(0.0)))
        cols.append(np.log1p(r['g'].to_numpy()))
    return np.column_stack(cols) @ WANN


def gru_direction(A, uid):
    """Delta z старого Gap-GRU из сохранённого чекпоинта, честного для A.

    Вызов воспроизводит regimes.py дословно: чекпоинт из weights/, маска
    ПРАВОВЫРОВНЕННАЯ и булева, третья координата prior центрирована.
    Любое отличие здесь молча даёт другое направление.
    """
    import torch
    from ecup.gapgru import GapGRUConfig, make_model, pick_device
    from ecup.tokens import TOKEN_FEATURES
    gi, ai = TOKEN_FEATURES.index('gap'), TOKEN_FEATURES.index('age')
    dev = pick_device()
    o = np.load(O / f'oofpm_a{A}.npz'); ouid = o['user_id']
    meta = np.load(TOK / f'meta_a{A}.npz')
    common, ti, oi = np.intersect1d(meta['user_id'], ouid, return_indices=True)
    X = np.load(TOK / f'x_a{A}.npy', mmap_mode='r'); L = meta['lengths'][ti]
    z0 = o['z0'][oi]; dis = o['z0_lgb'][oi] - o['z0_cb'][oi]
    ML = X.shape[1]
    m = make_model(GapGRUConfig(n_features=len(TOKEN_FEATURES) - 2, max_len=ML)).to(dev)
    ck = O / 'weights' / CK[A]
    if not ck.exists():
        ck = O / CK[A]
    sd = torch.load(ck, map_location=dev, weights_only=False)['model']
    miss, _ = m.load_state_dict(sd, strict=False)
    bad = [k for k in miss if not k.startswith(('factor', 'head_dp', 'head_dm'))]
    assert not bad, f'не загрузились веса: {bad[:5]}'
    m.eval(); out = []
    with torch.no_grad():
        for st in range(0, len(common), 4096):
            sl = slice(st, st + 4096)
            Xb = np.asarray(X[ti[sl]], dtype='float32'); Lb = L[sl]
            msk = np.arange(ML)[None, :] >= (ML - np.minimum(Lb, ML))[:, None]
            pr = np.stack([z0[sl] - z0.mean(), dis[sl],
                           np.log1p(Lb) - np.log1p(L).mean()], 1)
            T = lambda v, t=torch.float32: torch.as_tensor(v, dtype=t, device=dev)
            out.append(m(T(np.delete(Xb, [gi, ai], axis=2)), T(Xb[:, :, gi]),
                         T(Xb[:, :, ai]), T(msk, torch.bool), T(pr))[0]
                       .float().cpu().numpy().ravel())
    dz = np.concatenate(out)
    full = np.full(len(uid), np.nan)
    pos = {int(u): k for k, u in enumerate(uid)}
    for k, u in enumerate(common):
        j = pos.get(int(u))
        if j is not None:
            full[j] = dz[k]
    return full, float(np.isfinite(full).mean())


if __name__ == '__main__':
    df = load_panel()
    ref = np.load('artifacts/recon/d_annual_378.npy')
    o = np.load(O / 'oofpm_a378.npz')
    mine = annual(df, 378, o['user_id'])
    print(f'годовой фильтр на 378: corr {np.corrcoef(mine, ref)[0,1]:+.6f} · '
          f'std мой {mine.std():.5f} против сохранённого {ref.std():.5f}')
    dref = np.load(O / 'dz_a378.npz')
    g, cov = gru_direction(378, o['user_id'])
    gr = (pl.DataFrame({'user_id': o['user_id']})
          .join(pl.DataFrame({'user_id': dref['user_id'],
                              'v': dref['dz'].astype('float64')}),
                on='user_id', how='left')['v'].to_numpy())
    m = np.isfinite(gr) & (np.abs(g) > 0)
    print(f'направление GRU на 378: покрытие {cov:.4f} · пересечение {m.mean():.4f} · '
          f'corr {np.corrcoef(g[m], gr[m])[0,1]:+.6f} · '
          f'std мой {g[m].std():.5f} против сохранённого {gr[m].std():.5f}')
