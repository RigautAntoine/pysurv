import numpy as np

class AalenAdditive():
    """
    Aalen's additive regression model
    """
    
    def __init__(self, events, durations, X, entry_time = None):
        """
        Params:
            events (numpy.array): 0 or 1
            durations (numpy.array): time at which event happened or observation was censored 
            entry_time (numpy.array): time of entry into study (default is None - entry T is 0 for all)
            X (numpy.matrix): data matrix
        """
        self.events = events
        self.durations = durations
        self.X = X
        
        if entry_time is None:
            entry_time = np.zeros((len(self.events),))
        
        self.entry_time = entry_time
        
        self._fit(X, events, entry_time, durations)
    
    
    def _fit(self, X, events, entry_time, durations):
    
        n, p = X.shape
        ids = np.arange(len(X))

        # Matrix of interest
        matrix = np.vstack([ids, events, entry_time, durations, X.T]).T

        # Event times
        unique_event_times = np.unique(durations[events==1])
        unique_event_times.sort()
        T = len(unique_event_times)

        Y = np.zeros((T, n, p))
        I = np.zeros((T, n))
        # At each event time
        for j, t in enumerate(unique_event_times):
            # Get t
            risk_pool = matrix[(entry_time < t) & (durations >= t)] 
            Y[j, risk_pool[:,0].astype(int)] = risk_pool[:,4:]
            I[j, risk_pool[(risk_pool[:,1] == 1) & (risk_pool[:,3] == t),0].astype(int)] = 1

        A = np.zeros((T+1, p))
        cov_A = np.zeros((T+1, p, p))
        for i, t in enumerate(unique_event_times):
            try:
                X_t = np.dot(np.linalg.inv(np.dot(Y[i].T, Y[i])), Y[i].T)
            except:
                X_t = np.zeros((p, n))
            I_t = I[i,:]
            I_d_t = np.diag(I_t)
            A[i+1] = A[i] + np.dot(X_t, I_t)
            cov_A[i+1] = cov_A[i] + np.dot(X_t, I_d_t).dot(X_t.T)

        A = A[1:]
        cov_A = cov_A[1:]
            
        self.coefficients = A
        self.covars = cov_A
        self._unique_event_times = unique_event_times