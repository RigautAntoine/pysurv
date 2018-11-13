import numpy as np

class ExponentialUnivariate():
    """
    Implements parametric Exponential model for univariate data:
    S(t) = exp(-(lambda*t)),   lambda >0
    H(t) = lambda*t
    h(t) = lambda

    The probability of failure is the same in every time interval, i.e. hazard rate is constant.
    The constant hazard function is a consequence of the memoryless property of the exponential
    distribution: the distribution of the subject’s remaining survival time given that s/he has
    survived till time t does not depend on t.
    """
    
    def __init__(self, events, durations):
        self.events = events
        self.durations = durations
        
        self._fit(events, durations)
    
    def _fit(self, events, durations):
        D = events.sum()
        T = durations.sum()
        
        self._lambda = D / T
        self._lambda_variance = self._lambda / T
        self._max_duration = np.max(durations)
        
        self._timeline = np.linspace(0, self._max_duration, 100)
        self._survival = np.exp(-(self._lambda * self._timeline))
        self._cumhaz = self._lambda * self._timeline