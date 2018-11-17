import numpy as np
from scipy import stats


def extract_AIC(log_likelihood, p):
    return extract_criterion(log_likelihood, p, k=2)

def extract_BIC(log_likelihood, p, n):
    log_n = np.log(n)
    return extract_criterion(log_likelihood, p, k=log_n)
    #n = self._data.shape[0]
    #return criterion(n)

def extract_criterion(log_likelihood, p, k):
    """
    log_likelihood
    p: number of parameters
    k: parameter. k = 2 for AIC, k = log(n) for BIC where n is the number of observation
    """
    return -2 * log_likelihood + k * p
    #return -2 * self._loglik[1] + k * self._data.shape[1]

def chisquare_test(statistic, df):
    p_value = stats.chi2.sf(statistic, df)
    return p_value

def likelihood_ratio(main, nested):
    """
    main: CoxPH instance
    nested: CoxPH instance nested in main (less params)
    """
    from scipy.stats import chi2
    chisquare = -2 * nested._loglik[1] - (-2) * main._loglik[1] # Chi-square statistics
    df = main._data.shape[1] - nested._data.shape[1] # Numbers of degrees of freedoms
    p_value = chisquare_test(chisquare, df = df) # P-value
    print('Likelihood Ratio Test')
    print('Main: ' + main._formula)
    print('Nested: ' + nested._formula)
    print('Df = ' + str(df))
    print('Chi-square = {}. P-value = {}'.format(chisquare, p_value))


def inv_normal_cdf(p):

    def AandS_approximation(p):
        # Formula 26.2.23 from A&S and help from John Cook ;)
        # http://www.johndcook.com/normal_cdf_inverse.html
        c_0 = 2.515517
        c_1 = 0.802853
        c_2 = 0.010328

        d_1 = 1.432788
        d_2 = 0.189269
        d_3 = 0.001308

        t = np.sqrt(-2 * np.log(p))

        return t - (c_0 + c_1 * t + c_2 * t ** 2) / (1 + d_1 * t + d_2 * t * t + d_3 * t ** 3)

    if p < 0.5:
        return -AandS_approximation(p)
    else:
        return AandS_approximation(1 - p)