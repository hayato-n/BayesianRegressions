import numpy as np
from scipy import special
from tqdm import tqdm

from . import Updaters


class _baseLR(object):
    def __init__(self, y, X):
        self.y = np.array(y).flatten()
        self.X = np.array(X)

        self.N, self.D = self.X.shape
        assert self.N == len(self.y)

    def fit(self):
        raise NotImplementedError()

    def predict(self, X):
        return X @ self.weights_


class ARD(_baseLR):
    def __init__(self, y, X, a=[1e-2, 1e-4], b=[1e-2, 1e-4]):
        self.a = np.array(a)
        self.b = np.array(b)

        assert len(self.a) == 2
        assert len(self.b) == 2

        super().__init__(y, X)

    def fit(self, maxiter, batchsize=None):
        assert type(maxiter) is int
        assert maxiter > 0
        if batchsize is None:
            # Variational Inference
            batchsize = self.N
            self.updater = Updaters._ForgetUpdater()
        else:
            # Stochastic Variational Inference
            assert type(batchsize) is int
            assert batchsize > 0
            self.updater = Updaters.AdaptiveUpdater()

        self.batchsize = batchsize
        idx = np.random.choice(range(self.N), size=self.batchsize, replace=False)
        self._initialize(idx)

        self.ELBO = [self._calc_ELBO(idx)]
        with tqdm(total=maxiter) as pbar:
            pbar.set_description("ELBO={:.2f}".format(self.ELBO[-1]))

            for i in range(maxiter):
                pbar.update(1)
                idx = np.random.choice(
                    range(self.N), size=self.batchsize, replace=False
                )
                self._updateW(idx)
                self._updateARD(idx)
                self.ELBO.append(self._calc_ELBO(idx))

                pbar.set_description("ELBO={:.2f}".format(self.ELBO[-1]))

    def _initialize(self, idx):
        self.weights_ = np.zeros((self.D, 1))
        self.prec_ = np.prod(self.a)
        self.C = np.ones(self.D) * np.prod(self.b)

        self._xxT = self.X[..., None] @ self.X[:, None]
        self._yX = self.y[:, None] * self.X
        self._y2 = np.square(self.y)

        # self._xxTmean = np.mean(self._xxT, axis=0)
        # self._yXmean = np.mean(self._yX, axis=0)
        # self._y2mean = np.mean(self._y2, axis=0)

        self._lam_h = np.zeros((self.D))
        self._lam_hhT = np.zeros((self.D, self.D))

        self._sf_W = self._calc_sf_W(idx)
        self._sf_ARD = self._calc_sf_ARD(idx)
        self.updater.init_lr("W", self._sf_W)
        self.updater.init_lr("ARD", self._sf_ARD)

        self._updateW(idx)
        self._updateARD(idx)

    def _updateW(self, idx):
        sf = self._calc_sf_W(idx)
        self.updater.updatelr("W", self._sf_W, sf)
        self._sf_W = [self.updater("W", self._sf_W[i], sf[i]) for i in range(len(sf))]
        self._sf2paramW(self._sf_W, idx)

    def _updateARD(self, idx):
        sf = self._calc_sf_ARD(idx)
        self.updater.updatelr("ARD", self._sf_ARD, sf)
        self._sf_ARD = [
            self.updater("ARD", self._sf_ARD[i], sf[i]) for i in range(len(sf))
        ]
        self._sf2paramARD(self._sf_ARD)

    def _calc_sf_W(self, idx):
        # sufficient statistics
        return (
            -0.5 * self.N * np.mean(self._xxT[idx], axis=0),
            np.array(0.5 * self.N),
            -0.5 * self.N * np.mean(self._y2[idx]),
            self.N * np.mean(self._yX, axis=0),
        )

    def _calc_sf_ARD(self, idx):
        # sufficient statistics
        return (self._lam_hhT, np.array(0.5))

    def _sf2paramW(self, sf, idx):
        self.C_ = np.diag(self.C) - 2 * sf[0]
        C_w = sf[3]
        self.weights_ = np.linalg.solve(self.C_, C_w)
        # self.a_ = self.a + np.array([sf[1], -sf[2] - 0.5 * self.weights_ @ C_w])
        self.a_ = self.a + np.array(
            [
                sf[1],
                0.5
                * (
                    self.N
                    * np.mean(np.square(self.y[idx] - self.X[idx] @ self.weights_))
                    + self.weights_ @ np.diag(self.C) @ self.weights_
                ),
            ]
        )

        # mean fields
        self.prec_ = self.a_[0] / self.a_[1]
        self._log_prec_ = special.digamma(self.a_[0]) - np.log(self.a_[1])
        self._lam_h = self.prec_ * self.weights_
        self._C_inv = np.linalg.inv(self.C_)
        self._lam_hhT = (
            self.prec_ * self.weights_[:, None] @ self.weights_[None] + self._C_inv
        )

    def _sf2paramARD(self, sf):
        self.b_ = np.zeros((self.D, 2))
        self.b_ += self.b[None]
        self.b_ += np.concatenate(
            [sf[1] * np.ones((self.D, 1)), 0.5 * np.diag(sf[0])[:, None]], axis=1
        )

        # mean fields
        self.C = self.b_[:, 0] / self.b_[:, 1]
        self._logC = special.digamma(self.b_[:, 0]) - np.log(self.b_[:, 1])

    def _calc_ELBO(self, idx):
        # logp(y|X, h, lam) + logp(h, lam|C) - logq(h, lam)
        y = self.y[idx]
        X = self.X[idx]
        resid = y - X @ self.weights_
        xCinvx = (X[:, None, :] @ self._C_inv @ X[..., None]).reshape(len(idx))
        elbo = (
            -0.5
            * self.N
            * (np.log(2 * np.pi) + self.prec_ * np.mean(np.square(resid) + xCinvx))
            - 0.5 * np.linalg.slogdet(self.C_)[1]
            + 0.5 * self.D
        )
        elbo += -special.gammaln(self.a[0]) + self.a[0] * np.log(self.a[1])
        elbo -= -special.gammaln(self.a_[0]) + self.a_[0] * np.log(self.a_[1])
        elbo += -self.a[1] * self.prec_ + self.a_[0]

        # logp(C) - logq(C)
        elbo += self._calc_ard_elbo()
        return elbo

    def _calc_ard_elbo(self):
        # logp(C) - logq(C)
        elbo = np.sum(
            -special.gammaln(self.b[0])
            + self.b[0] * np.log(self.b[1])
            + special.gammaln(self.b_[:, 0])
            - self.b_[:, 0] * np.log(self.b_[:, 1])
        )
        return elbo

    def predict(self, X, return_var=False):
        if return_var:
            xVx = (X[:, None] @ self._C_inv[None] @ X[..., None]).flatten()
            return super().predict(X), (1 + xVx) * self.a_[1] / (self.a_[0] - 1)
        else:
            return super().predict(X)


class BayesRidge(ARD):
    def fit(self, maxiter, batchsize=None):
        assert type(maxiter) is int
        assert maxiter > 0
        if batchsize is None:
            # Variational Inference
            batchsize = self.N
            self.updater = Updaters._ForgetUpdater()
        else:
            # Stochastic Variational Inference
            assert type(batchsize) is int
            assert batchsize > 0
            self.updater = Updaters.AdaptiveUpdater()

        self.batchsize = batchsize
        idx = np.random.choice(range(self.N), size=self.batchsize, replace=False)
        self._initialize(idx)

        self.ELBO = [self._calc_ELBO(idx)]
        with tqdm(total=maxiter) as pbar:
            pbar.set_description("ELBO={:.2f}".format(self.ELBO[-1]))

            for i in range(maxiter):
                pbar.update(1)
                idx = np.random.choice(
                    range(self.N), size=self.batchsize, replace=False
                )
                self._updateW(idx)
                self._updatePenalty(idx)
                self.ELBO.append(self._calc_ELBO(idx))

                pbar.set_description("ELBO={:.2f}".format(self.ELBO[-1]))

    def _initialize(self, idx):
        self.weights_ = np.zeros((self.D, 1))
        self.prec_ = np.prod(self.a)
        self.C = np.ones(self.D) * np.prod(self.b)

        self._xxT = self.X[..., None] @ self.X[:, None]
        self._yX = self.y[:, None] * self.X
        self._y2 = np.square(self.y)

        # self._xxTmean = np.mean(self._xxT, axis=0)
        # self._yXmean = np.mean(self._yX, axis=0)
        # self._y2mean = np.mean(self._y2, axis=0)

        self._lam_h = np.zeros((self.D))
        self._lam_hhT = np.zeros((self.D, self.D))

        self._sf_W = self._calc_sf_W(idx)
        self._sf_Penalty = self._calc_sf_Penalty(idx)
        self.updater.init_lr("W", self._sf_W)
        self.updater.init_lr("Penalty", self._sf_Penalty)

        self._updateW(idx)
        self._updatePenalty(idx)

    def _updatePenalty(self, idx):
        sf = self._calc_sf_Penalty(idx)
        self.updater.updatelr("Penalty", self._sf_Penalty, sf)
        self._sf_Penalty = [
            self.updater("Penalty", self._sf_Penalty[i], sf[i]) for i in range(len(sf))
        ]
        self._sf2paramPenalty(self._sf_Penalty)

    def _calc_sf_Penalty(self, idx):
        # sufficient statistics
        return (self._lam_hhT, np.array(0.5))

    def _sf2paramPenalty(self, sf):
        self.b_ = self.b + 0.5 * np.array([self.D, np.diag(sf[0]).sum()])

        # mean fields
        self.C = self.b_[0] / self.b_[1] * np.ones(self.D)
        self._logC = special.digamma(self.b_[0]) - np.log(self.b_[1])

    def _calc_ard_elbo(self):
        # logp(C) - logq(C)
        elbo = (
            -special.gammaln(self.b[0])
            + self.b[0] * np.log(self.b[1])
            + special.gammaln(self.b_[0])
            - self.b_[0] * np.log(self.b_[1])
        )
        return elbo
