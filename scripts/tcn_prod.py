"""Боевой TCN на плотной решётке: обучение на 318/348/378, прогноз 408.

Маржинал поверх полного стека составил +0.000251 — в 3.3 раза больше
BTYD и в 1.8 раза больше нейронаправления, при ортогональной доле 0.904.

Боевой аналог выигравшей конфигурации: те же три среза, сдвинутые
вперёд на шаг, те же веса, лучшая эпоха из фолдового прогона (10).
"""
import sys, warnings, time; sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np, torch, polars as pl
from pathlib import Path
from torch import nn
from ecup.dense_tcn import TCNConfig, make_tcn

O = Path('artifacts/neural'); D = O / 'dense'
TR = [318, 348, 378]; WA = [0.5, 0.7, 1.0]; FIN = 408; EPOCHS = 10
cfg = TCNConfig(direct=True, epochs=EPOCHS)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A_M = 0.18


class Anchor:
    def __init__(self, A, with_y=True):
        self.X = np.load(D / f'x_a{A}.npy', mmap_mode='r')
        uid = np.load(D / f'meta_a{A}.npz')['user_id']
        f = O / f'oofpm_a{A}.npz'
        o = np.load(f if f.exists() else O / f'oof_a{A}.npz')
        idx = {int(u): k for k, u in enumerate(o['user_id'])}
        keep = np.array([k for k, u in enumerate(uid) if int(u) in idx])
        oi = np.array([idx[int(uid[k])] for k in keep])
        self.rows = keep; self.uid = uid[keep]
        self.z0 = o['z0'][oi].astype('float32')
        self.dis = ((o['z0_lgb'][oi] - o['z0_cb'][oi]).astype('float32')
                    if 'z0_lgb' in o.files else np.zeros(len(oi), 'float32'))
        self.z = (np.log1p(o['y'][oi]).astype('float32') if with_y and 'y' in o.files
                  else np.zeros(len(oi), 'float32'))
        self.act = np.asarray(self.X[keep][:, :, 0]).sum(1).astype('float32')
        self.prior = np.stack([self.z0, self.dis, np.log1p(self.act)], 1)
        print(f'  якорь {A}: {len(keep):,}', flush=True)

    def batch(self, sl):
        i = self.rows[sl]
        return (torch.as_tensor(np.asarray(self.X[i], dtype='float32'), device=dev),
                torch.as_tensor(self.prior[sl], device=dev),
                torch.as_tensor(self.z[sl], device=dev))


tr = [Anchor(a) for a in TR]; fin = Anchor(FIN, with_y=False)
cfg.init_bias = float(np.concatenate([d.z for d in tr]).mean())
model = make_tcn(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
mse = nn.MSELoss(); rng = np.random.default_rng(cfg.seed)
for ep in range(EPOCHS):
    model.train(); t0 = time.perf_counter(); tot = 0.0; nb = 0
    order = [(gi, s) for gi, d in enumerate(tr)
             for s in range(0, len(d.z), cfg.batch_size)]
    rng.shuffle(order)
    for gi, s in order:
        d = tr[gi]; sl = slice(s, min(s + cfg.batch_size, len(d.z)))
        x, pr, z = d.batch(sl)
        loss = mse(model(x, pr), z) * WA[gi]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tot += float(loss); nb += 1
    print(f'  эпоха {ep+1}/{EPOCHS} loss {tot/nb:.5f} · '
          f'{time.perf_counter()-t0:.0f}с', flush=True)

model.eval(); out = []
with torch.no_grad():
    for s in range(0, len(fin.z0), 512):
        sl = slice(s, min(s + 512, len(fin.z0)))
        x, pr, _ = fin.batch(sl)
        out.append(model(x, pr).float().cpu().numpy())
# Боевой уровень задаётся ручкой a_m, как во всех боевых направлениях
d_new = np.concatenate(out) + A_M - fin.z0
print(f'\nпрогноз на {FIN}: {len(d_new):,} · std направления {d_new.std():.5f}',
      flush=True)
ref = np.load(O / 'dz_prod_a408.npz')['user_id']
key = pl.DataFrame({'user_id': ref})
al = lambda u_, v_: key.join(pl.DataFrame({'user_id': u_, 'v': np.asarray(v_, 'float64')}),
                             on='user_id', how='left')['v'].to_numpy()
cen = lambda v: v - v.mean()
d = cen(np.nan_to_num(al(fin.uid, d_new)))
d_gru = np.nan_to_num(al(ref, np.load(O / 'dz_prod_a408.npz')['dz']))
lm = np.load(O / 'longmoney_prod_a408.npz'); d_life = np.nan_to_num(al(lm['user_id'], lm['d']))
gp = np.load(O / 'gruprod_dir_a408.npz'); d_g4 = np.nan_to_num(al(gp['user_id'], gp['d_raw']))
bp = np.load(O / 'btyd_prod_a408.npz'); d_bt = np.nan_to_num(al(bp['user_id'], bp['d']))
sys.path.insert(0, 'scripts')
from strong_base import annual
from ecup import load_panel
d_ann = annual(load_panel(), FIN, ref)
DM = np.column_stack([cen(d_gru), cen(d_ann), cen(d_life), cen(d_g4), cen(d_bt)])
b = np.linalg.lstsq(DM, d, rcond=None)[0]; dp = d - DM @ b
print(f'  ортогональная доля {dp.std()/d.std():.4f} · '
      f'corr GRU {np.corrcoef(d,d_gru)[0,1]:+.3f} · GRU-new {np.corrcoef(d,d_g4)[0,1]:+.3f} · '
      f'BTYD {np.corrcoef(d,d_bt)[0,1]:+.3f} · life {np.corrcoef(d,d_life)[0,1]:+.3f}')
np.savez_compressed(O / 'tcn_prod_a408.npz', user_id=ref, d=d, d_orth=dp)
torch.save({'model': model.state_dict()}, O / 'weights' / 'tcn_prod.pt')
print('сохранено tcn_prod_a408.npz', flush=True)
