import numpy as np
import scikits.bootstrap as sk_bootstrap

class Bootstrap:
    """
    UNFINISHED DONT USE
    Takes a list of values and calculates confidence interval. Possible for non-normal distributions.
    """
    def __init__(self, observations, B=19, n=-1, alpha=0.05, method='mean'):
        """
        :param observations: list of observations
        :param B: Number of repetitions. 19 for alpha=0.05, 99 for alpha=0.01
        :param n: Number of bootstrap samples generated per repetition. default equals to len(values)
        :return:
        """
        self.observations = observations
        self.B = B
        self.n = n if n > 0 else len(observations)
        self.alpha = alpha
        # initialise default values
        self.ci_lo = 0
        self.ci_hi = 0
        self.bootstrap_mean = 0



    def _bootstrap(self):
        # get B times n random samples with replacement
        bootstrap_samples = [np.random.choice(self.observations, size=self.n, replace=True) for i in range(self.B)]
        bootstrap_means = sorted([np.mean(sample) for sample in bootstrap_samples])
        self.bootstrap_mean = np.mean(bootstrap_means)
        self.ci_lo = bootstrap_means[round(self.B*self.alpha)]
        self.ci_hi = bootstrap_means[round(self.B*(1-self.alpha))-1]
        return self

        # get mean of means + CI
    def print(self):
        s = "{} [{}, {}]".format(self.bootstrap_mean, self.ci_lo, self.ci_hi)
        print(s)
        return s

class CSFSBootstrap:

    @staticmethod
    def get_ci(observations):
        ci = sk_bootstrap.ci(data=observations, statfunction=np.mean, n_samples=2000,)
        return ci

observations = [1, 3, 4, 5, 2, 3, 4, 1, 4, 3, 2, 1, 3, 3, 3,3 ,2 ]
bootstrap = Bootstrap(observations, B=2000)._bootstrap()
bootstrap.print()
print(np.mean(observations))

CI = sk_bootstrap.ci(data=observations, statfunction=np.mean, n_samples=19,)
print(CI)


