import numpy as np

import logging

# set logger (ref. https://www.mojirca.com/2019/12/get-started-python-logging.html)
loglevel = logging.INFO

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(loglevel)

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
handler.setLevel(loglevel)

logger.addHandler(handler)


class RobbinsMonroUpdater:
    """Parameter Updater with Robbins Monro Algorithm
    """

    def __init__(self, kappa, tau):
        """Parameter Updater with Robbins Monro Algorithm

        Parameters
        ----------
        kappa : float
            forgetting rate (0.5, 1]
        tau : float
            delay >= 0
        """
        self.kappa = float(kappa)
        self.tau = float(tau)

        self.t = dict()
        self.lr = dict()

    def __call__(self, name, old_params, new_params):
        lr = self.lr[name][-1]
        return (1 - lr) * old_params + lr * new_params

    def init_lr(self, name, params):
        self.t[name] = 0
        self.lr[name] = []

    def updatelr(self, name, old_params, new_params):
        self.t[name] += 1
        self.lr[name].append((self.t[name] + self.tau) ** (-self.kappa))


class _ForgetUpdater(RobbinsMonroUpdater):
    """just update parameter with new parameter
    """

    def __init__(self):
        super().__init__(0, 0)

    def updatelr(self, name, old_params, new_params):
        self.lr[name].append(1)


class AdaptiveUpdater(RobbinsMonroUpdater):
    """Parameter Updater with Adaptive Learning Rate Algorithm (Rajesh et al. 2013)
    """

    def __init__(self, init_sample_size=1, tol=1e-5):
        """Parameter Updater with Adaptive Learning Rate Algorithm (Rajesh et al. 2013)

        Parameters
        ----------
        init_sample_size : int, optional
            sample size to initilazise parameters, by default 1
        """
        self.init_sample_size = int(init_sample_size)
        self.lr = dict()
        self.g = dict()
        self.gTg = dict()
        self.tau = dict()

        self.tol = tol

    def init_lr(self, name, params):
        self.tau[name] = self.init_sample_size + self.tol
        self.lr[name] = []
        g = np.concatenate([param.flatten() for param in params])
        g = g.reshape((np.prod(g.shape), 1))
        self.g[name] = g.reshape((np.prod(g.shape), 1))
        self.gTg[name] = float(g.T @ g)

    def updatelr(self, name, old_params, new_params):
        g = np.concatenate([param.flatten() for param in new_params]) - np.concatenate(
            [param.flatten() for param in old_params]
        )
        g = g.reshape((np.prod(g.shape), 1))

        tau_inv = 1 / self.tau[name]
        self.g[name] = (1 - tau_inv) * self.g[name] + tau_inv * g
        self.gTg[name] = (1 - tau_inv) * self.gTg[name] + tau_inv * float(g.T @ g)

        try:
            self.lr[name].append(float(self.g[name].T @ self.g[name]) / self.gTg[name])
        except FloatingPointError:
            # avoid underflow
            logger.debug(
                "learning rate of {} was rounded to avoid underflow".format(name)
            )
            g = self.g[name]
            g[np.abs(g) < self.tol] = 0
            self.lr[name].append(float(g.T @ g) / self.gTg[name])

        self.tau[name] = self.tau[name] * (1 - self.lr[name][-1]) + 1 + self.tol
