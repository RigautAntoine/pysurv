import numpy as np
from scipy.optimize import minimize

def _partial_log_likelihood(betas, X, events, durations):
    """
    Assumes no ties in the data
    """
    unique_event_times = np.unique(durations[events==1])
    unique_event_times.sort()

    loglik = 0

    for i, t in enumerate(unique_event_times):

        loglik += X[(events==1) & (durations==t)].dot(betas)[0] - np.log(np.sum(np.exp(X[(durations >= t)].dot(betas))))

    return -loglik

def _efron_partial_log_likelihood(betas, X, events, durations):
    """
    Assumes no ties in the data
    """
    unique_event_times = np.unique(durations[events==1])
    unique_event_times.sort()

    loglik = 0

    for i, t in enumerate(unique_event_times):
        
        # Set of covariates with tied event time t
        tied_covariates = X[(events==1) & (durations==t)]
        m = tied_covariates.shape[0]
        tied_theta_sum = np.sum(np.exp(tied_covariates.dot(betas)))
      
        # Covariates at risk at time t
        atrisk_covariates = X[(durations>=t)]
        atrisk_theta_sum = np.sum(np.exp(atrisk_covariates.dot(betas)))
        
        loglik_t = np.sum(tied_covariates.dot(betas))
        
        for l in range(m):
            
            loglik_t -= np.log(atrisk_theta_sum - l/m * tied_theta_sum)
        
        loglik += loglik_t

    return -loglik

class CoxPH():
    """
    Implement the Cox's proportional hazards model
    
    TO-DO: implement the Efron method for solving ties
    https://en.wikipedia.org/wiki/Proportional_hazards_model
    """
    
    def __init__(self, events, durations, X):
        self.events = events
        self.durations = durations
        self.X = X
        self.betas = np.zeros(X.shape[1])
        self._fit()
        
    def _fit(self):
        
        self.betas = minimize(_efron_partial_log_likelihood, 
                            x0 = self.betas, 
                            args= (self.X, self.events, self.durations))['x']