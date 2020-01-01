"""
data.csv file data analysis
Cols : (SI units)
Time, Ax, Ay, Az, Absolute acceleration
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from typing import Tuple, Callable
from decimal import Decimal

g: float = 9.81


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.loadtxt("data.csv", skiprows=1, delimiter=",")
    _time, _ax, _ay, _az, _abs_a = (d[:, i] for i in range(d.shape[1]))
    _ind = np.array(list(x * y for x, y in zip(_time >= 1, _time < 585)))
    return _time[_ind] - 1, _ax[_ind], _ay[_ind]


def soft_map(func: Callable, tab: np.ndarray, cap) -> np.ndarray:
    n = len(tab)
    new = []
    for i in range(n):
        new.append(func(tab[max(0, int(i - cap / 2)): min(n, int(i + cap / 2))]))
    return np.array(new)


def kapa_det(_ay, _time):
    gy: np.ndarray = np.gradient(_ay, _time)
    ae_m2kt: np.ndarray = soft_map(max, _ay[gy == 0], 20) - g
    e_m2kt: np.ndarray = ae_m2kt/max(ae_m2kt)
    kt: np.ndarray = -1/2 * np.log(e_m2kt)[_time[gy == 0] < 300]
    t: np.ndarray = (_time[gy == 0])[_time[gy == 0] < 300].reshape(-1, 1)

    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(t, kt)
    k = reg.coef_[0]
    k_ = '%.2E' % Decimal(k)
    r2 = round(reg.score(t,kt),2)

    fig, _ax = plt.subplots()
    _ax.set_title(f"kapa = {k_}, R2 = {r2}", fontsize=16)
    _ax.plot(t, reg.predict(t), "-r")
    _ax.plot(t, kt, "+g")
    _ax.set_xlabel('t')
    _ax.set_ylabel('kt')
    fig.show()
    return k_, r2


def omega_det(_x, _time):
    sinwt: np.ndarray = _x[_time < 10]
    t: np.ndarray = _time[_time < 10]
    """
    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(t, sinwt)
    k = reg.coef_[0]
    k_ = '%.2E' % Decimal(k)
    r2 = round(reg.score(t, sinwt), 2)
    """
    fig, _ax = plt.subplots()
    # _ax.set_title(f"w = {w_}, R2 = {r2}", fontsize=16)
    # _ax.plot(t, reg.predict(t), "-r")
    _ax.plot(t, sinwt, "+g")
    _ax.set_xlabel('t')
    _ax.set_ylabel('sin(wt)')
    fig.show()
    # return w_, r2


if __name__ == "__main__":
    time, ax, ay = get_data()
    # print("k = {0}\tr2 = {1}".format(*kapa_det(ay, time)))
    # print("w = {0}\tr2 = {1}".format(*omega_det(ay, time)))
    omega_det(ay, time)
