import numpy as np
from scipy.integrate import trapz


def integ(li: np.ndarray, _t: np.ndarray) -> np.ndarray:
    return np.array([trapz(li[:i], x=_t[:i]) for i in range(len(li))])


def rebinet(li: np.ndarray, n) -> np.ndarray:
    new = []
    for i in range(len(li)):
        new.append(li[max(0, i-n):min(len(li),i+n)].mean())
    return np.array(new)


def rm_o(li: np.ndarray, n) -> np.ndarray:
    new = []
    for i in range(len(li)):
        new.append(li[i] - li[max(0, i-n):min(len(li), i+n)].mean())
    return np.array(new)


if __name__ == '__main__':
    from prog import get_data
    import matplotlib.pyplot as plt
    n = 10
    bo = 0.05
    v0 = 0
    ti, ax, ay = get_data()
    t = ti[ti < n]
    ax = rm_o(rebinet(ax[ti < n], 50), 20000)
    ay = rm_o(rebinet(ay[ti < n], 50), 20000)
    ind = np.array(list(x * y for x, y in zip(ax >= -bo, ax <= bo)))
    time = t[ind]
    delta = np.array([time[i] - time[max(0, i-1)] for i in range(len(time))])
    delta = 2*delta[delta > 0.2]
    print(f"w = {delta.mean()} +- {delta.std()}")
