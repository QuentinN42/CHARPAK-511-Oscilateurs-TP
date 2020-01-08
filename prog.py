"""
data.csv file data analysis
Cols : (SI units)
Time, Ax, Ay, Az, Absolute acceleration
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from typing import Tuple, Callable
from decimal import Decimal
from integration import rm_o, rebinet


g: float = 9.81


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.loadtxt("data.csv", skiprows=1, delimiter=",")
    _time, _ax, _ay, _az, _abs_a = (d[:, i] for i in range(d.shape[1]))
    _ind = np.array(list(x * y for x, y in zip(_time >= 2, _time < 585)))
    return _time[_ind] - 2, _ax[_ind], _ay[_ind]


def soft_map(func: Callable, tab: np.ndarray, cap) -> np.ndarray:
    n = len(tab)
    new = []
    for i in range(n):
        new.append(func(tab[max(0, int(i - cap / 2)): min(n, int(i + cap / 2))]))
    return np.array(new)


def kapa_det(_ay, _time, plot: bool = True, bo: float = 0.05):
    gy: np.ndarray = np.gradient(_ay, _time)
    ind = np.array(list(x * y for x, y in zip(gy >= -bo, gy <= bo)))
    t = _time[ind][_ay[ind] > 0]
    ae_m2kt: np.ndarray = _ay[ind][_ay[ind] > 0]
    e_m2kt: np.ndarray = ae_m2kt/max(ae_m2kt)
    kt: np.ndarray = -1 / 2 * np.log(e_m2kt)[t < 300]
    t = t[t < 300].reshape(-1, 1)

    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(t, kt)
    k = reg.coef_[0]
    k_ = '%.2E' % Decimal(k)
    r2 = round(reg.score(t, kt), 4)

    if plot:
        fig, _ax = plt.subplots()
        _ax.set_title(f"kapa = {k_}, R2 = {r2}", fontsize=16)
        _ax.plot(t, reg.predict(t), "-r")
        _ax.plot(t, kt, "+g")
        _ax.set_xlabel('t')
        _ax.set_ylabel('kt')
        fig.show()
    return k_, r2


def T_det(_x, _time, bo: float = 0.05, plot: bool = False):
    ind = np.array(list(x * y for x, y in zip(_x >= -bo, _x <= bo)))
    time = _time[ind]
    delta = np.array([time[i] - time[max(0, i - 1)] for i in range(len(time))])
    delta = delta[delta > 0.2][::2] + delta[delta > 0.2][1::2]

    if plot:
        plt.plot(_time, _x, "r")
        plt.plot(time, _x[ind], "ob")
        plt.show()

    return delta.mean(), delta.std()


if __name__ == "__main__":
    n = 5
    ti, ax, ay = get_data()
    t = ti
    ax = rm_o(rebinet(ax, 50), 20000)
    ay = rm_o(rebinet(ay, 50), 20000)
    # print(t.shape, ax.shape, ay.shape)
    # print("k = {0}\tr2 = {1}".format(*kapa_det(ax, t)))
    T, err = T_det(ax[ti < n], t[ti < n], bo=0.01)
    print(f"T = {T} +- {err}")
