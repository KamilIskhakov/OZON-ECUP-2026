"""Дискретные режимы активности: восемь состояний, residual-lookup с усадкой.

Смена ТИПА гипотезы, а не очередная обработка временной формы: пять
подряд закрытых веток были гладкими функциями истории.

Три неперекрывающихся блока по 30 дней дают бинарный паттерн присутствия
покупок. Никакого дробления по GMV-квинтилям и никаких дополнительных
величин: если восемь грубых состояний пусты, шестьдесят четыре дадут
in-sample шум, как уже было с когортной картой.
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
dev = pick_device(); df = load_panel(); LAM = 5000.0
Jann = np.arange(2, 11); wann = np.cos(2*np.pi*(-379.5+30*Jann.astype(float))/365)
wann = wann - wann.mean()
CK = {318:'gapgru_evt_ckpt_fold0.pt', 348:'gapgru_evt_ckpt_fold0.pt',
      378:'gapgru_evt_ckpt_fold1.pt'}

def prep(A):
    o = np.load(O/f'oof_a{A}.npz'); uid = o['user_id']
    e = np.log1p(o['y']) - o['z0']; u = pl.Series('user_id', uid)
    def has(lo, hi):
        t=(df.filter(pl.col('d').is_between(lo,hi)&pl.col('user_id').is_in(u)&(pl.col('gmv')>0))
             .group_by('user_id').agg(n=pl.len()))
        r=(pl.DataFrame({'user_id':uid}).join(t,on='user_id',how='left')
             .with_columns(pl.col('n').fill_null(0)).sort('user_id'))
        return (r['n'].to_numpy() > 0).astype(int)
    s = 4*has(A-89,A-60) + 2*has(A-59,A-30) + has(A-29,A)
    cols = []
    for j in Jann:
        t=(df.filter(pl.col('d').is_between(A-364+30*j, A-335+30*j)&pl.col('user_id').is_in(u))
             .group_by('user_id').agg(g=pl.col('gmv').sum()))
        r=(pl.DataFrame({'user_id':uid}).join(t,on='user_id',how='left')
             .with_columns(pl.col('g').fill_null(0.0)).sort('user_id'))
        cols.append(np.log1p(r['g'].to_numpy()))
    dann = np.column_stack(cols) @ wann
    meta = np.load(TOK/f'meta_a{A}.npz')
    common, ti, oi = np.intersect1d(meta['user_id'], uid, return_indices=True)
    X = np.load(TOK/f'x_a{A}.npy', mmap_mode='r'); L = meta['lengths'][ti]
    z0, dis = o['z0'][oi], o['z0_lgb'][oi]-o['z0_cb'][oi]
    m = make_model(GapGRUConfig(n_features=len(TOKEN_FEATURES)-2, max_len=192)).to(dev)
    sd = torch.load(O/'weights'/CK[A], map_location=dev, weights_only=False)['model']
    miss,_ = m.load_state_dict(sd, strict=False)
    assert not [k for k in miss if not k.startswith(('factor','head_dp','head_dm'))]
    m.eval(); out=[]
    with torch.no_grad():
        for st in range(0, len(common), 4096):
            sl=slice(st,st+4096); Xb=np.asarray(X[ti[sl]],dtype='float32'); Lb=L[sl]
            msk=np.arange(192)[None,:] >= (192-np.minimum(Lb,192))[:,None]
            pr=np.stack([z0[sl]-z0.mean(), dis[sl], np.log1p(Lb)-np.log1p(L).mean()],1)
            T=lambda v,t=torch.float32: torch.as_tensor(v,dtype=t,device=dev)
            out.append(m(T(np.delete(Xb,[gi,ai],axis=2)),T(Xb[:,:,gi]),T(Xb[:,:,ai]),
                         T(msk,torch.bool),T(pr))[0].float().cpu().numpy())
    return e, s, [np.concatenate(out), dann]

NAMES = {0:'000 нет покупок',1:'001 реактивация',2:'010 всплеск в середине',
         3:'011 недавно начал',4:'100 давно затух',5:'101 прерывисто',
         6:'110 недавно затух',7:'111 стабильно активен'}
DATA = {A: prep(A) for A in (318, 348, 378)}
print(f'{"состояние":<24} {"n на 318":>10} {"mu 318":>9} {"mu 348":>9} {"mu 378":>9}')
for st in range(8):
    row = []
    for A in (318, 348, 378):
        e, s, _ = DATA[A]; m_ = s == st
        row.append(e[m_].mean() - e.mean() if m_.sum() else np.nan)
    n0 = int((DATA[318][1] == st).sum())
    print(f'  {NAMES[st]:<22} {n0:>10,} ' + ' '.join(f'{v:>9.4f}' for v in row))
print(f'\n{"фолд":<20} {"alpha":>9} {"маржин":>10}')
sg = []
for tr, te in (([318], 348), ([318,348], 378)):
    es, ss = [], []
    for A in tr:
        e, s, _ = DATA[A]; es.append(e - e.mean()); ss.append(s)
    e_a, s_a = np.concatenate(es), np.concatenate(ss)
    mu = np.array([e_a[s_a==k].sum()/((s_a==k).sum()+LAM) for k in range(8)])
    e_t, s_t, ex = DATA[te]
    r = marginal_gain(e_t, mu[s_t], existing=ex)
    sg.append(np.sign(r['alpha_signed']))
    print(f'  {str(tr)+" -> "+str(te):<18} {r["alpha_signed"]:>+9.4f} {r["gain_marginal"]:>+10.5f}')
print(f'  знак {"совпал" if len(set(sg))==1 else "РАЗНЫЙ"}')
