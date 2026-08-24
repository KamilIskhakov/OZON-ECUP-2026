"""Прогноз боевого Gap-GRU на якоре 408 и сборка направления.

Скрипт обучения боевой режим не оценивает, поэтому инференс отдельный.
Вызов повторяет обучающий: маска правовыровненная и булева, prior из
z0, расхождения LGB-CB и центрированной длины.

Направление ортогонализуется против уже боевых направлений
(GRU, годовой фильтр, life), затем демеанируется. Коэффициент берётся
из ЧЕСТНОГО фолдового замера, а не подбирается здесь.
"""
import sys, warnings; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl, torch
from pathlib import Path
from ecup import load_panel
from ecup.gapgru import GapGRUConfig, make_model, pick_device
from ecup.tokens import TOKEN_FEATURES
from strong_base import annual

O = Path('artifacts/neural'); TOK = O / 'tokens384'; A = 408
dev = pick_device()
gi, ai = TOKEN_FEATURES.index('gap'), TOKEN_FEATURES.index('age')
meta = np.load(TOK / f'meta_a{A}.npz'); X = np.load(TOK / f'x_a{A}.npy', mmap_mode='r')
ofile = O / f'oofpm_a{A}.npz'
o = np.load(ofile if ofile.exists() else O / f'oof_a{A}.npz')
uid_o = o['user_id']
common, ti, oi = np.intersect1d(meta['user_id'], uid_o, return_indices=True)
L = meta['lengths'][ti]
z0 = o['z0'][oi].astype('float32')
dis = ((o['z0_lgb'][oi] - o['z0_cb'][oi]).astype('float32')
       if 'z0_lgb' in o.files else np.zeros(len(oi), 'float32'))
ML = X.shape[1]
m = make_model(GapGRUConfig(n_features=len(TOKEN_FEATURES) - 2, max_len=ML)).to(dev)
sd = torch.load(O / 'weights' / 'gruprod_prod.pt', map_location=dev,
                weights_only=False)['model']
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
print(f'прогноз на {A}: {len(common):,} пользователей · std {dz.std():.5f}')

ref = np.load(O / 'dz_prod_a408.npz')['user_id']
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(
    pl.DataFrame({'user_id': u_, 'v': np.asarray(v_, dtype='float64')}),
    on='user_id', how='left')['v'].to_numpy()
d_new = np.nan_to_num(al(common, dz))
d_gru = np.nan_to_num(al(ref, np.load(O / 'dz_prod_a408.npz')['dz']))
lm = np.load(O / 'longmoney_prod_a408.npz')
d_life = np.nan_to_num(al(lm['user_id'], lm['d']))
d_ann = annual(load_panel(), A, ref)
cen = lambda v: v - v.mean()
D = np.column_stack([cen(d_gru), cen(d_ann), cen(d_life)])
b = np.linalg.lstsq(D, cen(d_new), rcond=None)[0]
dp = cen(d_new) - D @ b
print(f'  ортогональная доля {dp.std()/cen(d_new).std():.4f} · '
      f'corr с GRU {np.corrcoef(d_new, d_gru)[0,1]:+.4f} · '
      f'годовым {np.corrcoef(d_new, d_ann)[0,1]:+.4f} · '
      f'life {np.corrcoef(d_new, d_life)[0,1]:+.4f}')
np.savez_compressed(O / 'gruprod_dir_a408.npz', user_id=ref,
                    d_raw=cen(d_new), d_orth=dp)
print(f'  сохранено gruprod_dir_a408.npz · std сырого {cen(d_new).std():.5f} · '
      f'ортогонального {dp.std():.5f}')
