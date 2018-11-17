import numpy as np
from scipy.optimize import minimize

def _negative_log_likelihood(lambda_rho, E, T):
    if any([x <= 0 for x in lambda_rho]):
        return np.inf
    _lambda, rho = lambda_rho
    ll = 0
    n = len(E)
    for i in range(n):
        
        ll += np.log(1./ (1. + (_lambda*T[i])**rho))
        ll += E[i] * np.log((rho*_lambda*(_lambda*T[i])**(rho-1)) / (1. + (_lambda*T[i])**rho))
        
    return -ll

def _negative_log_likelihood_multivariate(lambda_rho, E, T, X):
    
    beta, rho = lambda_rho[:-1], lambda_rho[-1]
    _lambda = np.exp(-X.dot(beta))
    
    if rho <= 0:
        return np.inf
    
    ll = 0
    n = len(E)
    for i in range(n):
        
        ll += np.log(1./ (1. + (_lambda[i]*T[i])**rho))
        ll += E[i] * np.log((rho*_lambda[i]*(_lambda[i]*T[i])**(rho-1)) / (1. + (_lambda[i]*T[i])**rho))
        
    return -ll

class LogLogistic():
    """
    Implements Multivariate case
    """
    def __init__(self, events, durations, X):
        self.events = events
        self.durations = durations
        self.X = X
        self.k = X.shape[1]
        self._fit()
    
    def _fit(self):
        lambda_rho = minimize(_negative_log_likelihood_multivariate, 
                            x0 = np.ones(self.k+1)*0.01, 
                            args= (self.events, self.durations, self.X))['x']
        
        self._lambda, self._rho =  lambda_rho[:-1], lambda_rho[-1]
    
class LogLogisticUnivariate():
    """
    Implements Log-logistic model:
    S(t) = 1 / (1 + (lambda*t)**rho)
    H(t) = log(1 + (lambda*t)**rho)
    h(t) = (rho*lambda*(lambda*t)**(rho-1)) / (1 + (lambda*t)**rho)
    
    Accomodates a unimodel function of hazard over time (hazard has a peak)
    
    """
    def __init__(self, events, durations):
        self.events = events
        self.durations = durations
        self._fit()
    
    def _fit(self):
        lambda_rho = minimize(_negative_log_likelihood, 
                            x0 = (1., 1.), 
                            args= (self.events, self.durations))['x']
        
        self._lambda, self._rho =  lambda_rho
        
        self._max_duration = self.durations.max()
        self._timeline = np.linspace(0, self._max_duration, 100)
        self._survival = 1. / (1. + (self._lambda * self._timeline)**self._rho)