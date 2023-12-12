import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class SIMULATE(object):
    """
    Simulation based on the Monte Carlo exercise from
    "Banbura, Marta and Modugno, Michele,
    Maximum Likelihood Estimation of Factor Models on Data Sets with Arbitrary Pattern of Missing Data
    (April 30, 2010). ECB Working Paper No. 1189,
    Available at SSRN: https://ssrn.com/abstract=1598302"
    Augmented with polynomial and sign factors.
    """

    def __init__(self, seed: int, n: int = 10, r: int = 1, poly_degree: int = 1, sign_features: int = 0,
                 rho: float = 0.7, alpha: float = 0.2, u: float = 0.1, tau: float = 0):
        """

        Args:
            seed: seed setting for replicability
            n: number of observable components
            r: number of common factors
            poly_degree: polynomial degree (1 linear)
            sign_features: whether to add features based on sign (the first "sign_features" features, if 0 than not)
            rho: parameter governing serial-correlation of the common factors
            alpha: parameter governing serial-correlation of the idiosyncratic components
            u: parameter governing the signal-to-noise ratio
            tau: parameter governing cross-correlation of the idiosyncratic components
        """
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.n = n
        self.r = r
        self.poly_degree = poly_degree
        self.sign_features = sign_features
        self.rho = rho
        self.alpha = alpha
        self.u = u
        self.tau = tau
        self.linear_f = None
        self.f = None

    def simulate(self, t_obs: int, portion_missings: float = 0.0) -> np.ndarray:
        """
        Method to simulate data from the DGP defined by the class.
        Args:
            t_obs: number of observations to simulate
            portion_missings: portion of missing data

        Returns:
            the simulated observable variables
        """
        # common factors
        u_t = self.rng.multivariate_normal(np.zeros(self.r), np.identity(self.r), t_obs)
        A = np.diag(self.rho * np.ones(self.r))
        f = np.zeros_like(u_t)
        for t in range(t_obs):
            if t > 0:
                f[t, :] = f[t - 1, :] @ A + u_t[t, :]
            else:
                f[t, :] = u_t[t, :]
        self.linear_f = f.copy()
        if self.poly_degree > 1:
            poly = PolynomialFeatures(self.poly_degree, include_bias=False)
            f = poly.fit_transform(f)
        if self.sign_features > 0:
            f = np.hstack((f, np.sign(f[:, :self.sign_features])))
        # loadings
        Lambda = self.rng.multivariate_normal(np.zeros(f.shape[1]), np.identity(f.shape[1]), self.n)
        # idio 
        beta = self.rng.uniform(self.u, 1 - self.u, self.n)
        gamma = np.zeros_like(beta)
        for i in range(self.n):
            gamma[i] = beta[i] / (1 - beta[i]) * 1 / (1 - self.alpha ** 2) * np.sum(Lambda[i, :] ** 2)
        phi = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                phi[i, j] = (self.tau ** (np.abs(i - j))) * (1 - self.alpha ** 2) * np.sqrt(gamma[i] * gamma[j])
        v_t = self.rng.multivariate_normal(np.zeros(self.n), phi, t_obs)
        D = np.diag(self.alpha * np.ones(self.n))
        eps = np.zeros_like(v_t)
        for t in range(t_obs):
            if t > 0:
                eps[t, :] = eps[t - 1, :] @ D + v_t[t, :]
            else:
                eps[t, :] = v_t[t, :]
        # gen observables
        x = f @ Lambda.T + eps
        self.f = f
        # insert missings
        if portion_missings > 0:
            assert 0 < portion_missings < 1, "Portion Missing should be between 0 and 1"
            n_missings = int(t_obs * self.n * portion_missings)
            list_missing_loc = set()
            while len(list_missing_loc) < n_missings:
                _row, _col = self.rng.choice(t_obs, 1)[0], self.rng.choice(self.n, 1)[0]
                if (_row, _col) not in list_missing_loc:
                    list_missing_loc.add((_row, _col))
                    x[_row, _col] = np.nan
        return x

    def evaluate(self, f_hat: np.ndarray, f_true: np.ndarray = None) -> float:
        """
        Method to compute the trace R^2 between f_hat and f_true. If f_true is not provided then the factors stores as
        attributes are used.
        Args:
            f_hat: estimated factors
            f_true: true factors (if None, then "self.f" is used)

        Returns:
            the trace R^2 score
        """
        if f_true is None:
            f_true = self.f
        precision_score = np.trace(f_true.T @ f_hat @ np.linalg.pinv(f_hat.T @ f_hat) @ f_hat.T @ f_true) / np.trace(
            f_true.T @ f_true)
        return precision_score


if __name__ == "__main__":
    SIM = SIMULATE(seed=1)
    x = SIM.simulate(100)
    print(x.shape)
    print(SIM.f.shape)
