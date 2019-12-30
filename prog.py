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

g: float = 9.81


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.loadtxt("data.csv", skiprows=1, delimiter=",")
    _time, _ax, _ay, _az, _abs_a = (d[:, i] for i in range(d.shape[1]))
    return _time[_time >= 1], _ax[_time >= 1], _ay[_time >= 1]


def soft_map(func: Callable, tab: np.ndarray, cap) -> np.ndarray:
    n = len(tab)
    new = []
    for i in range(n):
        new.append(func(tab[max(0, int(i - cap / 2)): min(n, int(i + cap / 2))]))
    return np.array(new)


if __name__ == "__main__":
    time, ax, ay = get_data()
    gy: np.ndarray = np.gradient(ay, time)
    Ae_mkt: np.ndarray = soft_map(max, ay[gy == 0], 20) - g
    e_mkt: np.ndarray = Ae_mkt/max(Ae_mkt)
    kt: np.ndarray = -np.log(e_mkt)[time[gy == 0] < 300]
    t: np.ndarray = (time[gy == 0])[time[gy == 0] < 300].reshape(-1, 1)

    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(t, kt)
    k = reg.coef_[0]

    fig, _ax = plt.subplots()
    _ax.plot(t, reg.predict(t), "-r")
    _ax.plot(t, kt, "+g")
    _ax.set_xlabel('t')
    _ax.set_ylabel('kt')
    fig.show()

    print(f"kapa = {k}")
    print(f" R2  = {reg.score(t,kt)}")
