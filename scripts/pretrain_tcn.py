"""Этап A для TCN: тот же простой надзор, что сработал у Gap-GRU.

Расширенный претрейн с 23 целями уже пробовался и дал ХУДШУЮ
согласованность коэффициентов между фолдами (0.388 и 0.559 против
0.350 и 0.360 у простого). Поэтому здесь воспроизводится исходный
набор факторных целей, а не новый исследовательский проект:

    freq      число заказов за 7 и 30 дней, покупочных дней за 30
    amount    log(1+GMV) за 30 среди покупавших, средний чек
    intent    факт покупки на горизонтах 3, 7, 14, 30
    activity  активных дней за 7 и 30

Учителя-GBDT нет: для любого среза T с наблюдаемым будущим все цели
считаются напрямую из панели. Пуржинг: срез допустим при T + 30 <= A,
где A — якорь подбора alpha соответствующего фолда. Отступ равен
ГОРИЗОНТУ, а не шагу нарезки — на смешении этих величин уже была
поймана утечка.
"""
import sys, warnings, gc, time; sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np, polars as pl, torch
from pathlib import Path
from torch import nn
from ecup import load_panel
from ecup.dense_tcn import TCNConfig, DenseTCN

O = Path('artifacts/neural'); D = O / 'dense'
SLICES = [258, 288, 318]          # те же срезы, что и в этапе B
HEADS = {'freq': 3, 'amount': 2, 'intent': 4, 'activity': 2}
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def targets(df, A, uid):
    """Цели горизонта 30 дней после A, считаются прямо из панели."""
    w = df.filter(pl.col('d').is_between(A + 1, A + 30))
    agg = (w.group_by('user_id').agg(
        o7=(pl.col('to_ord') * (pl.col('d') <= A + 7)).sum(),
        o30=pl.col('to_ord').sum(),
        bd30=((pl.col('gmv') > 0)).sum(),
        g30=pl.col('gmv').sum(),
        a7=((pl.col('d') <= A + 7)).sum(),
        a30=pl.len(),
        c3=((pl.col('gmv') > 0) & (pl.col('d') <= A + 3)).sum(),
        c7=((pl.col('gmv') > 0) & (pl.col('d') <= A + 7)).sum(),
        c14=((pl.col('gmv') > 0) & (pl.col('d') <= A + 14)).sum()))
    j = pl.DataFrame({'user_id': uid}).join(agg, on='user_id', how='left')
    g = lambda c: j[c].fill_null(0).to_numpy().astype('float32')
    o30, g30, bd30 = g('o30'), g('g30'), g('bd30')
    return {
        'freq': np.stack([np.log1p(g('o7')), np.log1p(o30), np.log1p(bd30)], 1),
        'amount': np.stack([np.log1p(g30),
                            np.log1p(g30 / np.maximum(o30, 1.0))], 1),
        'intent': np.stack([(g('c3') > 0), (g('c7') > 0), (g('c14') > 0),
                            (g30 > 0)], 1).astype('float32'),
        'activity': np.stack([np.log1p(g('a7')), np.log1p(g('a30'))], 1),
    }


class Multi(nn.Module):
    """Тот же энкодер, но с факторными головами вместо одной поправки."""
    def __init__(self, cfg):
        super().__init__()
        self.enc = DenseTCN(cfg)
        ch = cfg.channels
        self.heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(ch * 3, 64), nn.ReLU(), nn.Linear(64, n))
            for k, n in HEADS.items()})

    def feats(self, x):
        h = self.enc.inp(x.transpose(1, 2))
        for b in self.enc.blocks:
            h = b(h)
        return torch.cat([h[:, :, -1], h[:, :, -30:].mean(-1), h.mean(-1)], 1)

    def forward(self, x):
        f = self.feats(x)
        return {k: hd(f) for k, hd in self.heads.items()}


if __name__ == '__main__':
    df = load_panel()
    cfg = TCNConfig(direct=True)
    model = Multi(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    data = []
    for A in SLICES:
        X = np.load(D / f'x_a{A}.npy', mmap_mode='r')
        uid = np.load(D / f'meta_a{A}.npz')['user_id']
        data.append((X, targets(df, A, uid)))
        print(f'  срез {A}: {len(uid):,}', flush=True)
    rng = np.random.default_rng(42); EP = 8; BS = 256
    for ep in range(EP):
        model.train(); t0 = time.perf_counter(); tot = 0.0; nb = 0
        order = [(gi, s) for gi, (X, _) in enumerate(data)
                 for s in range(0, len(X), BS)]
        rng.shuffle(order)
        for gi, s in order:
            X, T = data[gi]; sl = slice(s, min(s + BS, len(X)))
            x = torch.as_tensor(np.asarray(X[sl], dtype='float32'), device=dev)
            out = model(x)
            loss = 0.0
            for k in HEADS:
                y = torch.as_tensor(T[k][sl], device=dev)
                loss = loss + (bce(out[k], y) if k == 'intent' else mse(out[k], y))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss); nb += 1
        print(f'  эпоха {ep+1}/{EP} loss {tot/nb:.5f} · '
              f'{time.perf_counter()-t0:.0f}с', flush=True)
    (O / 'weights').mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.enc.state_dict()}, O / 'weights' / 'tcn_pretrain.pt')
    print('сохранено tcn_pretrain.pt', flush=True)
