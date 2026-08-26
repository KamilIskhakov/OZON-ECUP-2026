"""TCN на плотной решётке, прямое предсказание z.

ПЕРВАЯ ПОПЫТКА обучала на остатке e = z - z0 и схлопнулась в ноль:
std(dz) выросла до 0.005 на двадцатом шаге и вернулась к нулю к сотому.
Диагностика сняла подозрение с архитектуры — входы содержательны,
активации проходят все восемь блоков, вход головы различается между
примерами со std 1.25.

Объяснение в журнале: «локальный пилот без претрейна дал
std(dz) = 0.041 и потолок +0.00010 — направление почти ортогонально
остатку». Обучение последовательностной модели на ОСТАТКЕ с нуля уже
проверялось и не работает; Gap-GRU заработал лишь с этапом A.
Математически: оптимум c = E[s e]/1.05, и при почти нулевой корреляции
представления с остатком верный минимум — тождественный ноль.

Претрейн на плотной решётке стоит часы. Дешевле учить сам z: сигнал
там сильный и заведомо обучаемый, а направлением станет z_TCN - z0 —
разность двух полноценных прогнозов, как CatBoost против LightGBM.

Протокол ровно тот, что доказал себя для Gap-GRU и дал там +0.00135:
обучение на 258/288/318 с весами 0.5/0.7/1.0, подбор alpha на 348,
единственная настоящая оценка — 378. Никаких архитектурных переборов.

Цель — остаток честной OOF-базы e = z - z0. Голова с нулевой
инициализацией, поэтому на первом шаге прогноз тождественно равен
базовому.

Шлюз задан ДО запуска: Delta_378 > 3e-4 при положительном 348.
"""
import sys, warnings, gc, time; sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np, torch
from pathlib import Path
from torch import nn
from ecup.dense_tcn import TCNConfig, make_tcn

O = Path('artifacts/neural'); D = O / 'dense'
TR = [258, 288, 318]; WA = [0.5, 0.7, 1.0]; AL, TE = 348, 378
cfg = TCNConfig()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'устройство {dev}', flush=True)


class Anchor:
    def __init__(self, A):
        self.X = np.load(D / f'x_a{A}.npy', mmap_mode='r')
        m = np.load(D / f'meta_a{A}.npz'); uid = m['user_id']
        o = np.load(O / f'oofpm_a{A}.npz')
        idx = {int(u): k for k, u in enumerate(o['user_id'])}
        keep = np.array([k for k, u in enumerate(uid) if int(u) in idx])
        oi = np.array([idx[int(uid[k])] for k in keep])
        self.rows = keep
        self.z = np.log1p(o['y'][oi]).astype('float32')
        self.z0 = o['z0'][oi].astype('float32')
        self.dis = (o['z0_lgb'][oi] - o['z0_cb'][oi]).astype('float32')
        self.act = np.asarray(self.X[keep][:, :, 0]).sum(1).astype('float32')
        self.uid = uid[keep]
        self.prior = np.stack([self.z0 - self.z0.mean(), self.dis,
                               np.log1p(self.act) - np.log1p(self.act).mean()], 1)
        print(f'  якорь {A}: {len(keep):,}', flush=True)

    def batch(self, sl):
        i = self.rows[sl]
        return (torch.as_tensor(np.asarray(self.X[i], dtype='float32'), device=dev),
                torch.as_tensor(self.prior[sl], device=dev),
                torch.as_tensor(self.z[sl], device=dev),
                torch.as_tensor(self.z0[sl], device=dev))


tr = [Anchor(a) for a in TR]; da = Anchor(AL); dt = Anchor(TE)
model = make_tcn(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
n_par = sum(p.numel() for p in model.parameters())
print(f'параметров {n_par:,}', flush=True)
mse = nn.MSELoss()
# Диагностика живости: если std(dz) не растёт за первые шаги, модель
# не учится, и это надо ловить сразу, а не через двенадцать эпох.
with torch.no_grad():
    x0, p0, _, _ = tr[0].batch(slice(0, 256))
    print(f'std(dz) при инициализации {float(model(x0, p0).std()):.6f}', flush=True)


@torch.no_grad()
def predict(d):
    model.eval(); out = []
    for s in range(0, len(d.z), 512):
        sl = slice(s, min(s + 512, len(d.z)))
        x, pr, _, _ = d.batch(sl)
        out.append(model(x, pr).float().cpu().numpy())
    return np.concatenate(out)


best = {'gain': -1e9, 'ep': 0, 'state': None}
rng = np.random.default_rng(cfg.seed)
for ep in range(cfg.epochs):
    model.train(); t0 = time.perf_counter(); tot = 0.0; nb = 0
    order = [(gi, s) for gi, d in enumerate(tr)
             for s in range(0, len(d.z), cfg.batch_size)]
    rng.shuffle(order)
    for gi, s in order:
        d = tr[gi]; sl = slice(s, min(s + cfg.batch_size, len(d.z)))
        x, pr, z, z0 = d.batch(sl)
        zh = model(x, pr) + z0        # база как смещение, а не как цель
        loss = mse(zh, z) * WA[gi]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tot += float(loss); nb += 1
        if ep == 0 and nb in (20, 100):
            with torch.no_grad():
                print(f'    шаг {nb}: std(dz) {float(model(x0, p0).std()):.6f}',
                      flush=True)
    if (ep + 1) % 2 == 0:
        dz = predict(da); e = da.z - da.z0
        Dv = float((dz ** 2).mean()); C = float((e * dz).mean())
        a = C / max(Dv, 1e-12)
        g = float(e.std()) - float((e - a * dz).std())
        print(f'  эпоха {ep+1}/{cfg.epochs} loss {tot/nb:.5f} · '
              f'якорь {AL}: выигрыш {g:+.5f} · alpha {a:+.4f} · '
              f'std(dz) {dz.std():.4f} · {time.perf_counter()-t0:.0f}с', flush=True)
        if g > best['gain']:
            best = {'gain': g, 'ep': ep + 1,
                    'state': {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
    else:
        print(f'  эпоха {ep+1}/{cfg.epochs} loss {tot/nb:.5f} · '
              f'{time.perf_counter()-t0:.0f}с', flush=True)

if best['state'] is not None:
    model.load_state_dict(best['state'])
    print(f'восстановлена эпоха {best["ep"]} (выигрыш {best["gain"]:+.5f})', flush=True)
dz_a = predict(da); e = da.z - da.z0
alpha = float((e * dz_a).mean()) / max(float((dz_a ** 2).mean()), 1e-12)
dz_t = predict(dt); et = dt.z - dt.z0
base = float(et.std()); got = float((et - alpha * dz_t).std())
print(f'\nна {AL} (подбор alpha): выигрыш {best["gain"]:+.5f} · alpha {alpha:+.4f}')
print(f'на {TE} (оценка):   {base:.5f} -> {got:.5f}  выигрыш {base-got:+.5f}')
print(f'перенос: {100*(base-got)/max(best["gain"],1e-9):.0f}%')
np.savez_compressed(O / 'tcn_dz_a378.npz', user_id=dt.uid, dz=dz_t, z0=dt.z0, z=dt.z)
torch.save({'model': model.state_dict(), 'epoch': best['ep']},
           O / 'weights' / 'tcn_fold.pt')
print(f'\nшлюз: Delta_378 > 3e-4 · '
      f'{"ПРОЙДЕН" if base-got > 3e-4 else "не пройден"}', flush=True)
