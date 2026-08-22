"""TabM в production-режиме на боевом якоре, в двух версиях сразу.

Обучаются ДВЕ модели одним прогоном: v1 воспроизводит смещённую цель
(вычитание условного l_plus), прошедшую двухфолдовый шлюз, v2 — цель
с безусловным уровнем. На лидерборд идёт v1: измеряется именно то
направление, которое шлюз пропустил.
"""
import sys, warnings, gc, time; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from ecup import (SplitConfig, load_panel, build_anchor, build_training_set,
                  to_matrix, anchor_weights)
from ecup.dataset import anchor_offsets
from ecup.tabm import TabMConfig, make_tabm
from ecup.gapgru import pick_device
import torch
from torch import nn

df = load_panel(); O = Path('artifacts/neural'); dev = pick_device()
sp = SplitConfig(max_history=300, with_state=True)
an = sp.refit_anchors()
Xd, y, aid, lv = build_training_set(df, an, sp, None, verbose=False)
X, feats = to_matrix(Xd); del Xd; gc.collect()
w = anchor_weights(aid); last = lv[max(an)]
_, l_plus = anchor_offsets(aid, lv)              # условное — цель v1
l_uncond = np.array([lv[a].l for a in aid])      # безусловное — цель v2
fin = build_anchor(df, sp.final_anchor, sp, None, with_target=False)
Xte, _ = to_matrix(fin.X, feats)
mu, sd = X.mean(0), X.std(0) + 1e-6
Xn = np.clip(((X - mu)/sd).astype('float32'), -8, 8)
Xt = np.clip(((Xte - mu)/sd).astype('float32'), -8, 8)
print(f'якоря {an}\nстрок {len(y):,} · тест {len(Xt):,} · признаков {X.shape[1]}', flush=True)

for tag, off in (('v1', l_plus), ('v2', l_uncond)):
    zt = np.log1p(y) - off
    cfg = TabMConfig(n_features=X.shape[1], seed=42)
    m = make_tabm(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    nb = len(y)//cfg.batch_size + 1
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs*nb)
    rng = np.random.default_rng(cfg.seed); T = lambda v: torch.as_tensor(v, device=dev)
    print(f'\n=== {tag} · цель z - {"l_plus" if tag=="v1" else "l"} ===', flush=True)
    for ep in range(cfg.epochs):
        t0, tot = time.perf_counter(), 0.0
        idx = rng.permutation(len(y))
        for s in range(0, len(idx), cfg.batch_size):
            b = idx[s:s+cfg.batch_size]
            loss = ((m(T(Xn[b])) - T(zt[b].astype('float32'))[:, None])**2 *
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
        for s in range(0, len(Xt), 8192):
            out.append(m(T(Xt[s:s+8192])).float().cpu().numpy())
    P = np.concatenate(out); z = np.clip(P.mean(1) + last.l, 0, None)
    print(f'  {tag}: среднее {z.mean():.4f} · доля нулей {(z<1e-9).mean():.4f}', flush=True)
    np.savez_compressed(O/f'tabm_prod_{tag}_a{sp.final_anchor}.npz',
                        z=z, uid=fin.X['user_id'].to_numpy())
    del m; gc.collect(); torch.cuda.empty_cache() if dev.type=='cuda' else None
print('\nготово', flush=True)
