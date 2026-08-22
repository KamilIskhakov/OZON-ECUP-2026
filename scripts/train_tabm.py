"""TabM прямой регрессией на log1p(y). Два честных временных фолда.

Никакого hurdle, residual teacher и новых признаков: проверяется ровно
одно — даёт ли третье семейство модели декоррелированную ошибку.
"""
import sys, warnings, gc, time, json; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from ecup import (SplitConfig, load_panel, build_anchor, build_training_set,
                  to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.tabm import TabMConfig, make_tabm
from ecup.gapgru import pick_device
from ecup.directions import marginal_gain
import torch
from torch import nn

df = load_panel(); O = Path('artifacts/neural'); dev = pick_device()
for TE, NTR in ((348, 4), (378, 6)):
    sp = SplitConfig(max_history=300, n_train_anchors=NTR, with_state=True,
                     val_anchor=TE) if 'val_anchor' in SplitConfig.__dataclass_fields__ \
         else SplitConfig(max_history=300, n_train_anchors=NTR, with_state=True)
    an = [a for a in sp.train_anchors() if a + 30 <= TE][-NTR:]
    Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
    X, feats = to_matrix(Xd); del Xd; gc.collect()
    w = anchor_weights(aid); last = lv[max(an)]
    val = build_anchor(df, TE, sp, None); Xva, _ = to_matrix(val.X, feats)
    z = np.log1p(val.y)
    # БЕЗУСЛОВНЫЙ уровень якоря, как в DirectGBDT. anchor_offsets отдаёт
    # l_plus — условное среднее по покупавшим (~4.2), оно нужно голове
    # регрессии hurdle, которая учится только на y>0. Для прямой модели
    # его вычитание сделало бы цель непокупателя равной -4.2 вместо нуля.
    lvl = np.array([lv[a].l for a in aid])
    zt = np.log1p(y) - lvl
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - mu)/sd).astype('float32'); Xvn = ((Xva - mu)/sd).astype('float32')
    Xn = np.clip(Xn, -8, 8); Xvn = np.clip(Xvn, -8, 8)
    print(f'\n=== оценка на {TE} · якоря {an} · строк {len(y):,} ===', flush=True)
    cfg = TabMConfig(n_features=X.shape[1], seed=42)
    m = make_tabm(cfg).to(dev)
    print(f'  параметров {sum(p.numel() for p in m.parameters()):,} · членов {cfg.k}', flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    nb = len(y)//cfg.batch_size + 1
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs*nb)
    rng = np.random.default_rng(cfg.seed); T = lambda v: torch.as_tensor(v, device=dev)
    for ep in range(cfg.epochs):
        t0, tot = time.perf_counter(), 0.0
        idx = rng.permutation(len(y))
        for s in range(0, len(idx), cfg.batch_size):
            b = idx[s:s+cfg.batch_size]
            pred = m(T(Xn[b]))
            # каждый член учится на общей цели; декорреляция идёт от адаптеров
            loss = ((pred - T(zt[b].astype('float32'))[:, None])**2 *
                    T(w[b].astype('float32'))[:, None]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if sch.last_epoch+1 < sch.total_steps: sch.step()
            tot += float(loss.item())
        if (ep+1) % 10 == 0:
            print(f'  эпоха {ep+1}/{cfg.epochs} loss {tot/nb:.5f} '
                  f'{time.perf_counter()-t0:.0f}с', flush=True)
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(Xvn), 8192):
            out.append(m(T(Xvn[s:s+8192])).float().cpu().numpy())
    P = np.concatenate(out); z_tabm = np.clip(P.mean(1) + last.l, 0, None)
    print(f'  TabM standalone shape {(z-z_tabm).std():.5f} · '
          f'разброс членов {P.std(1).mean():.4f}', flush=True)
    np.savez_compressed(O/f'tabm_{TE}.npz', z=z_tabm, members=P.astype('float32'),
                        uid=val.X['user_id'].to_numpy())
    o = np.load(O/f'oofpm_a{TE}.npz')
    if TE == 378:
        d16 = np.load(O/'dz_a378.npz')['dz']; dann = np.load('/tmp/d_annual_378.npy')
        E = np.load('/tmp/cb_ens2.npz')
        zl = np.mean([E[k] for k in E.files if k.startswith('lgb')],0)
        zc = np.mean([E[k] for k in E.files if k.startswith(('cb_','brd','dpw'))],0)
        # ТОЧНЫЙ аналог v22/v23, а не усечённый: обе поправки, с теми же
        # коэффициентами, что ушли в сабмит. Без annual-члена база была
        # 1.67905 вместо 1.67753, и направление TabM мерилось не от того,
        # поверх чего оно реально ляжет в production.
        base = (0.4*zl + 0.6*zc + 0.35*(d16-d16.mean())
                + 0.0104*(dann-dann.mean()))
        ex = [d16, dann]
    else:
        base = o['p0']*o['m0']; ex = []
    e = z - base
    r = marginal_gain(e, z_tabm - base, existing=ex)
    print(f'  база shape {e.std():.5f} · rho(ошибок) '
          f'{np.corrcoef(e, z - z_tabm)[0,1]:+.4f}', flush=True)
    print(f'  направление TabM: alpha {r["alpha_signed"]:+.4f} · '
          f'маржинально {r["gain_marginal"]:+.5f}', flush=True)
    # аналитический оптимум вместо сетки: lam* = (s_b^2 - rho s_b s_T) /
    # (s_b^2 + s_T^2 - 2 rho s_b s_T), где s — std ошибок, а не прогнозов
    eT = z - z_tabm; sb, sT = e.std(), eT.std(); rho = np.corrcoef(e, eT)[0,1]
    lam = (sb*sb - rho*sb*sT) / (sb*sb + sT*sT - 2*rho*sb*sT)
    print(f'    lam* {lam:+.4f} · shape в оптимуме '
          f'{(e - lam*(e-eT)).std():.5f} (было {sb:.5f})', flush=True)

    # кривая по числу членов внутреннего ансамбля: если она уже вышла на
    # плато к K=32, внешние сиды почти ничего не добавят. Усредняем по
    # случайным подмножествам, чтобы не мерить порядок членов.
    rs = np.random.default_rng(0); K = P.shape[1]
    print('    члены ансамбля:', end=' ')
    for k in (1, 2, 4, 8, 16, K):
        s = np.mean([(z - np.clip(P[:, rs.choice(K, k, replace=False)].mean(1)
                                  + last.l, 0, None)).std()
                     for _ in range(16 if k < K else 1)])
        print(f'K={k} {s:.5f}', end='  ')
    print(flush=True)
    del m, X, Xva; gc.collect()
