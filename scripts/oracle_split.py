"""Куда уходит ошибка: детекция покупки или её величина.

Четыре прогноза на квадрате оракулов. Текущий p*m; идеально знаем факт
покупки, но не сумму; идеально знаем сумму, но не факт; оба идеальны.
Поскольку log(1+0) = 0, при обоих оракулах ошибка ровно ноль, поэтому
Shapley-доли складываются в полную MSE.

Оговорка: оракул использует реализовавшееся будущее, поэтому это верхняя
граница, включающая случайность поведения. Но если одна сторона не даёт
почти ничего даже с оракулом, усиливать её точно незачем.
"""
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import numpy as np
for A in (348, 378):
    o = np.load(f'artifacts/neural/oofpm_a{A}.npz')
    y = o['y']; z = np.log1p(y); I = (y > 0).astype(float)
    p, m = o['p0'], o['m0']
    mse = lambda v: float(np.mean((z - v)**2))
    M00, M10, M01 = mse(p*m), mse(I*m), mse(p*z)
    Sp = 0.5*((M00 - M10) + M01); Sm = 0.5*((M00 - M01) + M10)
    print(f'=== якорь {A} · {len(y):,} пользователей · доля покупателей {I.mean():.3f} ===')
    print(f'  текущий p*m           RMSLE {np.sqrt(M00):.5f}')
    print(f'  оракул детекции I*m   RMSLE {np.sqrt(M10):.5f}  (снимает {M00-M10:.4f} MSE)')
    print(f'  оракул суммы  p*z     RMSLE {np.sqrt(M01):.5f}  (снимает {M00-M01:.4f} MSE)')
    print(f'  доли Шепли: детекция {Sp/M00:.1%} · величина {Sm/M00:.1%}')
    e = z - p*m
    print(f'  ошибка: непокупатели {np.mean(e**2*(1-I))/M00:.1%} · '
          f'покупатели {np.mean(e**2*I)/M00:.1%}')
    pos = I > 0
    q = np.searchsorted(np.quantile(z[pos], [.25,.5,.75,.95]), z[pos])
    sh = [float(np.mean(e[pos][q==k]**2)*(q==k).mean()*pos.mean()/M00) for k in range(5)]
    print(f'  внутри покупателей по квартилям z: ' +
          ' '.join(f'{v:.1%}' for v in sh) + '  (последний — верхние 5%)')
    print('  калибровка p по децилям:')
    dq = np.searchsorted(np.quantile(p, np.linspace(0,1,11)[1:-1]), p)
    rows = [(float(p[dq==k].mean()), float(I[dq==k].mean())) for k in range(10)]
    print('    p̂:   ' + ' '.join(f'{a:.2f}' for a,_ in rows))
    print('    факт:' + ' '.join(f'{b:.2f}' for _,b in rows))
    mq = np.searchsorted(np.quantile(m[pos], np.linspace(0,1,6)[1:-1]), m[pos])
    print('  калибровка m среди покупателей по квинтилям:')
    print('    m̂:   ' + ' '.join(f'{float(m[pos][mq==k].mean()):.2f}' for k in range(5)))
    print('    факт:' + ' '.join(f'{float(z[pos][mq==k].mean()):.2f}' for k in range(5)))
    print()
