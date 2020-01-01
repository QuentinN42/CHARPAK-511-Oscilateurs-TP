import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from typing import Callable
from decimal import Decimal
from scipy.optimize import curve_fit
from prog import g, get_data
from scipy.integrate import trapz


def integ(li: np.ndarray) -> np.ndarray:
    return np.array([trapz(li[:i]) for i in range(len(li))])


def rebinet(li: np.ndarray, n) -> np.ndarray:
    new = []
    for i in range(len(li)):
        new.append(li[max(0, i-n):min(len(li),i+n)].mean())
    return np.array(new)


def simple(t, r, theta_0, k, w, phi):
    x = (
            g
            - r
            * theta_0 ** 2
            * k
            * np.exp(-2 * k * t)
            * (k * np.cos(w * t + phi) + w * np.sin(w * t + phi)) ** 2
    )
    y = (
            r
            * theta_0
            * np.exp(-k * t)
            * (
                    (k ** 2 - w ** 2 - g / r) * np.cos(w * t + phi)
                    + 2 * k * w * np.sin(w * t + phi)
            )
    )
    # return np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1).astype(float)
    return y


class Simple:
    def __init__(self, r=None, theta_0=None, k=None, w=None, phi=None):
        self.fitted = False
        self.r = r
        self.theta_0 = theta_0
        self.k = k
        self.w = w
        self.phi = phi
        self.d = [r, theta_0, k, w, phi]
        self.nb_args = self.d.count(None)

    def _f(self, t, *args):
        it = iter(args)
        return simple(t, *[e if e is not None else next(it) for e in self.d])

    def __call__(self, t):
        if self.fitted:
            return self._f(t, self.args)
        else:
            raise ReferenceError("Call the fit first")

    def r2(self, t, a):
        return 1 - np.var(self(t) - a) / np.var(a)

    def fit(self, t, a):
        self.fitted = True
        self.args, pcov = curve_fit(simple, t, a, p0=[], maxfev=50000)


if __name__ == '__main__':
    vt0 = 0
    vr0 = 0
    ti, OMppt, OMppr = get_data()
    OMppt -= OMppt.mean()
    OMpt = integ(OMppt) + vt0
    plt.plot(ti, OMpt)
    plt.show()
    """
    pred = np.array([0.35, -15/180*np.pi, 3.3*10**-3, 7, 4])
    for max_time in range(10, 603, 10):
        print(max_time)
        args, pcov = curve_fit(simple, ti[ti < max_time], -ay[ti < max_time], p0=pred)
        print(args)
        pred = args
    """
    # final = [-2.93850172e-01  1.12427178e+00  4.99939834e-03  5.59400491e+00, 4.72693554e+00]
