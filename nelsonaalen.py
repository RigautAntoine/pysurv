import numpy as np
import pandas as pd
import matplotlib.pylab as plt


class NelsonAalen():
    """
    The Nelson–Aalen estimator is a non-parametric estimator of the cumulative 
    hazard rate function in case of censored data
    
    Handles left and right-censoring
    
    TO-DO: variance and confidence interval, and strata
    """
    
    def __init__(self, events, durations, entry_time=None):
        """
        Params:
            events (numpy.array): 0 or 1
            durations (numpy.array): time at which event happened or observation was censored 
            entry_time (numpy.array): time of entry into study (default is None - entry T is 0 for all)
        """
        self.events = events
        self.durations = durations
        
        if entry_time is None:
            entry_time = np.zeros((len(self.events),))
        
        self.entry_time = entry_time
        self._fit(events, durations, entry_time)
        
    def _fit(self, events, durations, entry_time):
    
        unique_event_times = np.unique(durations[events==1])
        unique_event_times.sort()
        n = len(unique_event_times)
        
        # Number of unique observed failure times
        n = len(unique_event_times)
        # Risk pool where value i correspond to at-risk objects at unique_event_times[i]
        risk = np.zeros((n,)) 
        # Failures at unique_event_times[i]
        failures = np.zeros((n,)) 
        
        for i, t in enumerate(list(unique_event_times)):
            risk[i] = np.sum((entry_time < t) & (durations >= t))
            failures[i] = np.sum((events == 1) & (durations == t))
            
        lifetable = pd.DataFrame({'at-risk': risk, 'failures':failures}, index=unique_event_times)
        lifetable['cumhaz'] = (failures / risk).cumsum()
        lifetable['survival'] = np.exp(-lifetable['cumhaz'])
        
        self._lifetable = lifetable
    
    def summary(self):
        '''
        Returns the life table
        Time => Nb at risk => Nb of events => Survival => VarSur => CIs => Hazard Rate => Cumlative
        '''
        return self._lifetable
    
    def plot_cumhaz(self,**kwargs):
        
        ax = plt.step(x = self._lifetable.index, y = self._lifetable['cumhaz'], **kwargs) 
        #ax.set_ylim(0, 1)
        #ax.set_xlim(0)
        
        return ax