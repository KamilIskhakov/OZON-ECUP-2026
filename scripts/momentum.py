"""Ускорение и затухание: контрасты по неперекрывающимся блокам.

Другой вопрос, чем годовая ось: не «какая форма у траектории», а
«пользователь сейчас ускоряется или затухает». Разность логов
L_i - L_j = log((1+Q_i)/(1+Q_j)) сокращает уровень автоматически —
именно тот inductive bias, который дал годовой фильтр.

Двенадцать признаков заданы заранее, перебора окон нет. Базис на обоих
якорях симметричен: нейронаправление честной модели плюс годовой фильтр.
"""
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np, polars as pl, torch
from pathlib import Path
from ecup import load_panel
from ecup.gapgru import GapGRUConfig, make_model, pick_device
from ecup.tokens import TOKEN_FEATURES
from ecup.directions import marginal_gain
O, TOK = Path('artifacts/neural'), Path('artifacts/neural/tokens')
gi, ai = TOKEN_FEATURES.index('gap'), TOKEN_FEATURES.index('age')
dev = pick_device(); df = load_panel()
QTY = {'gmv': pl.col('gmv').sum(), 'ord': pl.col('to_ord').sum(),
       'buy': (pl.col('gmv')>0).sum()}
Jann = np.arange(2, 11); wann = np.cos(2*np.pi*(-379.5+30*Jann.astype(float))/365)
wann = wann - wann.mean()

def neural_dir(A, ckpt):
    meta = np.load(TOK/f'meta_a{A}.npz'); oof = np.load(O/f'oof_a{A}.npz')
    common, ti, oi = np.intersect1d(meta['user_id'], oof['user_id'], return_indices=True)
    X = np.load(TOK/f'x_a{A}.npy', mmap_mode='r'); L = meta['lengths'][ti]
    z0, dis = oof['z0'][oi], oof['z0_lgb'][oi]-oof['z0_cb'][oi]
    m = make_model(GapGRUConfig(n_features=len(TOKEN_FEATURES)-2, max_len=192)).to(dev)
    sd = torch.load(O/'weights'/ckpt, map_location=dev, weights_only=False)['model']
    miss,_ = m.load_state_dict(sd, strict=False)
    assert not [k for k in miss if not k.startswith(('factor','head_dp','head_dm'))]
    m.eval(); out=[]
    with torch.no_grad():
        for s in range(0, len(common), 4096):
            sl=slice(s,s+4096); Xb=np.asarray(X[ti[sl]],dtype='float32'); Lb=L[sl]
            msk=np.arange(192)[None,:] >= (192-np.minimum(Lb,192))[:,None]
            pr=np.stack([z0[sl]-z0.mean(), dis[sl], np.log1p(Lb)-np.log1p(L).mean()],1)
            T=lambda v,t=torch.float32: torch.as_tensor(v,dtype=t,device=dev)
            out.append(m(T(np.delete(Xb,[gi,ai],axis=2)),T(Xb[:,:,gi]),T(Xb[:,:,ai]),
                         T(msk,torch.bool),T(pr))[0].float().cpu().numpy())
    return common, np.concatenate(out)

def prep(A, ckpt):
    o = np.load(O/f'oof_a{A}.npz'); uid = o['user_id']
    e = np.log1p(o['y']) - o['z0']; u = pl.Series('user_id', uid)
    def blk(lo, hi):
        t=(df.filter(pl.col('d').is_between(lo,hi)&pl.col('user_id').is_in(u))
             .group_by('user_id').agg(**QTY))
        r=(pl.DataFrame({'user_id':uid}).join(t,on='user_id',how='left')
             .with_columns(pl.exclude('user_id').fill_null(0)).sort('user_id'))
        return {q: np.log1p(r[q].to_numpy().astype('float64')) for q in QTY}
    Lk = [blk(A-30*(k+1)+1, A-30*k) for k in range(6)]
    F, N = [], []
    for q in QTY:
        L0,L1,L2,L3,L4,L5 = (Lk[k][q] for k in range(6))
        F += [L0-L1, (L0+L1)/2-(L2+L3)/2, (L0+L1+L2)/3-(L3+L4+L5)/3, L0-2*L1+L2]
        N += [f'{q}_m1', f'{q}_m2', f'{q}_m3', f'{q}_acc']
    # годовой фильтр на общей опоре
    cols = []
    for j in Jann:
        lo, hi = A-364+30*j, A-335+30*j
        t=(df.filter(pl.col('d').is_between(lo,hi)&pl.col('user_id').is_in(u))
             .group_by('user_id').agg(g=pl.col('gmv').sum()))
        r=(pl.DataFrame({'user_id':uid}).join(t,on='user_id',how='left')
             .with_columns(pl.col('g').fill_null(0.0)).sort('user_id'))
        cols.append(np.log1p(r['g'].to_numpy()))
    dann = np.column_stack(cols) @ wann
    cu, dn = neural_dir(A, ckpt)
    assert (cu == uid).all() or len(cu) == len(uid)
    return e, np.column_stack(F), N, [dn, dann]

CK = {348: 'gapgru_evt_ckpt_fold0.pt', 378: 'gapgru_evt_ckpt_fold1.pt',
      318: 'gapgru_evt_ckpt_fold0.pt'}
LAM = 30.0
GRP = {'все 12': slice(0,12), 'только GMV': slice(0,4), 'orders+buydays': slice(4,12)}
print(f'{"группа":<16} {"фолд":<18} {"alpha":>9} {"маржин":>10}')
for nm, sl in GRP.items():
    row = []
    for tr, te in (([318], 348), ([318,348], 378)):
        Xs, ys = [], []
        for A in tr:
            e, X, N, _ = prep(A, CK[A]); Xs.append(X[:, sl]); ys.append(e-e.mean())
        X = np.vstack(Xs); y = np.concatenate(ys)
        mu, sd_ = X.mean(0), X.std(0)+1e-9; Xz = (X-mu)/sd_
        coef = np.linalg.solve(Xz.T@Xz + LAM*np.eye(Xz.shape[1]), Xz.T@y)
        e_t, Xt, _, ex = prep(te, CK[te])
        d = ((Xt[:, sl]-mu)/sd_) @ coef
        r = marginal_gain(e_t, d, existing=ex)
        row.append((r['alpha_signed'], r['gain_marginal']))
        print(f'  {nm if te==348 else "":<14} {str(tr)+" -> "+str(te):<18} '
              f'{r["alpha_signed"]:>+9.4f} {r["gain_marginal"]:>+10.5f}')
    s = [np.sign(a) for a,_ in row]
    print(f'  {"":<14} знак {"совпал" if len(set(s))==1 else "РАЗНЫЙ"}')
