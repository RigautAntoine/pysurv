import numpy as np
import pandas as pd
import matplotlib.pylab as plt

class KaplanMeier():
    """
    Non-parametric estimator of the survival function
    for non- or right-censored data.
    
    TO-DO: Strata, confidence interval
    """
    
    def __init__(self, events, durations):
        """
        Params:
            events (numpy.array): 0 or 1
            durations (numpy.array): time at which event happened or observation was censored 
        """
        self._fitted = False
        self.events = events
        self.durations = durations
        
        self._fit(events, durations)
        
        
    def _fit(self, events, durations):
        
        unique_event_times = np.unique(durations[events==1])
        unique_event_times.sort()
        
        # Number of unique observed failure times
        n = len(unique_event_times)
        # Risk pool where value i correspond to at-risk objects at unique_event_times[i]
        risk = np.zeros((n,)) 
        # Failures at unique_event_times[i]
        failures = np.zeros((n,)) 
        
        for i, t in enumerate(list(unique_event_times)): 
            risk[i] = np.sum(durations >= t)
            failures[i] = np.sum((events == 1) & (durations == t))
        
        lifetable = pd.DataFrame({'at-risk': risk, 'failures':failures}, index=unique_event_times)
        lifetable['survival'] = np.cumprod((risk - failures)/risk)
        lifetable['cumhaz'] = -np.log(lifetable['survival'])
        
        self._lifetable = lifetable
        self.fitted = True
        
    def summary(self):
        '''
        Returns the life table
        Time => Nb at risk => Nb of events => Survival => VarSur => CIs => Hazard Rate => Cumlative
        '''
        return self._lifetable
    
    def plot_survival(self,**kwargs):
        
        ax = plt.step(x = self._lifetable.index, y = self._lifetable['survival'], **kwargs) 
        #ax.set_ylim(0, 1)
        #ax.set_xlim(0)
        
        return ax