"""
data.csv file data analysis
Cols : (SI units)
Time, Ax, Ay, Az, Absolute acceleration
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from typing import Tuple


def get_data() -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    d = np.loadtxt("data.csv", skiprows=1, delimiter=',')
    _time, _ax, _ay, _az, _abs_a = (d[:, i] for i in range(d.shape[1]))
    return _time[_time >= 1]-1, _ax[_time >= 1], _ay[_time >= 1]


if __name__ == '__main__':
    time, ax, ay = get_data()
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(time, ax, '+r')
    axs[1].plot(time, ay, '+g')
    fig.show()
