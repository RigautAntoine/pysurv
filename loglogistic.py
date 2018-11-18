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
        self.unique_event_times = np.unique(self.durations[self.events==1])
        self.unique_event_times.sort()
        self._fit()
    
    def _fit(self):
        self.res = minimize(_negative_log_likelihood_multivariate, 
                            x0 = np.ones(self.k+1)*0.01, 
                            args= (self.events, self.durations, self.X))
        
        lambda_rho = self.res['x']
        # Standard errors aren't good here
        self.standard_errors = np.sqrt(np.diag(self.res['hess_inv']))
        
        self._lambda, self._rho =  lambda_rho[:-1], lambda_rho[-1]
    
    def predict_hazards(self, X):
        return np.exp(-X.dot(self._lambda))
    
    def predict_cumulative_hazard(self, X):
        """
        Return the cumulative hazard function of each sample
        """
        
        timeline = np.tile(self.unique_event_times, (X.shape[0], 1))
        
        return pd.DataFrame(np.log(1. + (timeline.T * self.predict_hazards(X))**self._rho), 
                         index = self.unique_event_times)
    
    def predict_survival(self, X):
        """
        Return the survival function of each sample
        """
        
        return 1. / np.exp(self.predict_cumulative_hazard(X))
    
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