import numpy as np
from scipy.optimize import minimize

def _negative_log_likelihood(lambda_rho, E, T):
    if any([x <= 0 for x in lambda_rho]):
        return np.inf
    _lambda, rho = lambda_rho
    ll = 0
    n = len(E)
    for i in range(n):
        
        ll += -(_lambda*T[i])**rho
        ll += E[i] * np.log(rho * _lambda * (_lambda*T[i])**(rho-1))
        
    return -ll
    

class WeibullUnivariate():
    """
    Implements parametric Weibull accelerated failure time model:
    
    S(t) = exp(-(lambda*t)**rho),   lambda >0, rho > 0,
    H(t) = (lambda*t)**rho
    h(t) = rho*lambda*(lambda*t)**(rho-1)
    
    With rho the shape parameter.

    The Weibull distribution extends the exponential distribution to allow constant, increasing, 
    or decreasing hazard rates.  We can see that, depending on whether the shape parameter is 
    greater than or less than 1, the hazard can increase or decrease with increasing time.
    
    rho > 1. Hazard increase over time
    rho = 1. Hazard constant (Exponential)
    rho < 1. Hazard dicreases over time

    Goal is to estimate lambda, (rho), cumulative_hazard and survival
    
    Weibull is a generalization of the exponential model, where rho = 1
    
    In the multivariate case, rho is held constant across observation 
    while lambda is reparameterized in terms of predictor variables and covariate coefficients
    """
    def __init__(self, events, durations):
        self.events = events
        self.durations = durations
        self._fit()
    
    def _fit(self):
        lambda_rho = minimize(_negative_log_likelihood, 
                            x0 = (1, 1), 
                            args= (self.events, self.durations))['x']
        
        self._lambda, self._rho =  lambda_rho
        
        self._max_duration = self.durations.max()
        self._timeline = np.linspace(0, self._max_duration, 100)
        self._survival = np.exp(-(self._lambda * self._timeline)**self._rho)