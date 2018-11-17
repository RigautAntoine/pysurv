import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from .metrics import _concordance_index

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
    
    TO-DO: 
    Standard Errors: DONE
    Estimation of baseline hazard: DONE
    Coefficient plots
    Prediction: 
    Concordance index: DONE
    """
    
    def __init__(self, events, durations, X):
        self.events = events
        self.durations = durations
        self.X = X
        self.betas = np.zeros(X.shape[1])
        self._fit()
        
    def _fit(self):
        
        self.res = minimize(_efron_partial_log_likelihood, 
                            x0 = self.betas, 
                            args= (self.X, self.events, self.durations))
        self.betas = self.res['x']
        self.standard_errors = np.sqrt(np.diag(self.res['hess_inv']))
        
        self._compute_baseline_hazard()
    
    def predict_partial_hazards(self, data):
        return np.exp(data.dot(self.betas))
    
    def concordance(self):
        
        durations = self.durations
        events = self.events
        predicted = -self.predict_partial_hazards(self.X)
        
        cidx = _concordance_index(durations, predicted, events)
        self.cidx = cidx
        
        return self.cidx
    
    def _compute_baseline_hazard(self):
        """
        https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
        """
        unique_event_times = np.unique(self.durations[self.events==1])
        unique_event_times.sort()
        
        n=len(unique_event_times)
        h0 = np.zeros(n)
        
        for i, T in enumerate(unique_event_times):
            h0[i] = np.sum((self.events == 1) & (self.durations == T)) / np.sum(np.exp(self.X[self.durations >= T].dot(self.betas)))
            
        self.baseline_hazard = h0
        self.unique_event_times = unique_event_times
        self.cumulative_baseline_hazard = h0.cumsum()
    
    def predict_log_partial_hazards(self, X):
        """Equivalent to linear_predictors in R"""
        return X.dot(self.betas)
    
    def predict_partial_hazards(self, X):
        """Equivalent to risk in R"""
        return np.exp(self.predict_log_partial_hazards(X))
    
    def predict_median_lifetime(self, X):
        """
        Predict median lifetime
        """
        pass
    
    def predict_cumulative_hazard(self, X):
        """
        Return the cumulative hazard function of each sample
        """
        
        cumhaz = np.tile(self.cumulative_baseline_hazard, (X.shape[0], 1))
        
        return pd.DataFrame(cumhaz.T * self.predict_partial_hazards(X), 
                         index = self.unique_event_times)
    
    def predict_survival(self, X):
        """
        Return the survival function of each sample
        """
        
        return np.exp(-self.predict_cumulative_hazard(X))
    
    def coefplot(self):
        
        plt.errorbar(x=self.betas,yerr=self.standard_errors*1.96)