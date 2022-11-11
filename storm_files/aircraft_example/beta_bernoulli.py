
import numpy as np
from scipy import stats
import scipy.special as sc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


class BetaBernoulli(object):
    """
    Cleaned version of the class written in conjugate_prior package on github.

    If mode is provided beta is calculated as a function of alpha and rounded.
    """
    __slots__ = ['alpha', 'beta']

    def __init__(self, successes, failures=None, mode=None):
        self.alpha = successes
        if failures is not None and mode is None:
            self.beta = failures
        elif failures is None and mode is not None:
            # we cast to int to keep interpretability
            self.beta = int((self.alpha - 1 - (self.alpha - 2) * mode) / mode)
        else:
            raise ValueError('Either failures or mode must be provided.\n' \
                f'Current values are: failures = {failures}, mode = {mode}.')

    def update(self, successes, failures):
        self.alpha += successes
        self.beta += failures

    def pdf(self, x):
        return stats.beta.pdf(x, self.alpha, self.beta)

    def cdf(self, x):
        return stats.beta.cdf(x, self.alpha, self.beta)

    def ppf(self, q):
        """
        Percent point function (inverse of cdf — percentiles) at q.
        Aka Value at Risk.

        :param q: lower tail probability
        :return: quantile corresponding to the lower tail probability q.
        """
        return stats.beta.ppf(q, self.alpha, self.beta)

    def value_at_risk(self, q):
        """
        :param q: lower tail probability
        """
        return self.ppf(q)

    def conditional_value_at_risk(self, q):
        """
        # This is E(X|X >= x_alpha) conditioned to the interval [x_alpha, 1].
        # conditional = True -> divided by [cdf(ub) - cdf(lb)] = 1 - x_alpha
        # Check eq.(3) in https://www.ise.ufl.edu/uryasev/files/2011/11/VaR_vs_CVaR_INFORMS.pdf
        # if func in expect is None then f(x) = x -> expected value
        """
        value_at_risk = self.value_at_risk(q)
        cvar = stats.beta.expect(lambda x: x,
                                 args=(self.alpha, self.beta),
                                 lb=value_at_risk,
                                 ub=1,
                                 conditional=True)
        return cvar

    def expected_value(self):
        """
        Expected value E(X). 
        This is equal to:
        stats.beta.expect(lambda x: x, args=(alpha, beta), lb=0, ub=1)
        """
        return self.alpha / (self.alpha + self.beta)

    def posterior(self, l, u):
        if l > u:
            return 0.0
        return self.cdf(u) - self.cdf(l)

    def mean(self, n=1):
        return self.alpha * n / (self.alpha + self.beta)

    @property
    def mode(self):
        """
        Maximum A Posteriori (MAP)
        The MAP of a Bernoulli distribution with a Beta prior is the mode
        of the Beta posterior. The mode of a distribution is the value that
        maximizes the probability mass function (if discrete) or probability
        density function (if continuous). For alpha + beta > 2:
        """
        return (self.alpha - 1)/(self.alpha + self.beta - 2)

    def plot(self, l=0.0, u=1.0, alpha=0.5, label=None, **options):
        x = np.linspace(u, l, 1001)
        y = stats.beta.pdf(x, self.alpha, self.beta)
        y = y / y.sum()
        plt.plot(x, y, label=label, **options)
        plt.fill_between(x, y, np.zeros(1001), alpha=alpha, **options)
        plt.xlim((l, u))

    def predict(self, t, f, log=False):
        a = self.alpha
        b = self.beta
        log_pmf = (sc.gammaln(t + f + 1) + sc.gammaln(t + a) + sc.gammaln(f + b) + sc.gammaln(a + b)) - \
                  (sc.gammaln(t + 1) + sc.gammaln(f + 1) + sc.gammaln(a) + sc.gammaln(b) + sc.gammaln(t + f + a + b))
        if log:
            return log_pmf
        return np.exp(log_pmf)

    def sample(self, output_parameter=False):
        """
        :param bool output_parameter: True to sample from the distribution.
        """
        p = np.random.beta(self.alpha, self.beta)
        if output_parameter:
            return p
        return int(np.random.random() < p)

