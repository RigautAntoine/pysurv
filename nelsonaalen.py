import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from .stats import inv_normal_cdf

class NelsonAalen():
    
    def __init__(self, events, durations, alpha=0.95, strata=None):
        
        self.kms = []
        
        if strata is None:
            self.kms.append(NelsonAalenFitter(events, durations, label='', alpha=alpha))
        else:
            stratas = np.unique(strata)
            for s in stratas:
                m = (strata == s)
                self.kms.append(NelsonAalenFitter(events[m], durations[m], label=s, alpha=alpha))
                
    def summary(self):
        return [km.summary() for km in self.kms]
    
    def plot(self):
        
        ax = plt.figure().add_subplot(111)
        
        for km in self.kms:
            km.plot(ax=ax)
            
        
        ax.set_ylim(0)
        ax.set_xlim(0)
        ax.set_xlabel('Timeline')
        
        plt.legend(loc='best')

class NelsonAalenFitter():
    """
    The Nelson–Aalen estimator is a non-parametric estimator of the cumulative 
    hazard rate function in case of censored data
    
    Handles left and right-censoring
    
    TO-DO: variance and confidence interval, and strata
    """
    
    def __init__(self, events, durations, label, entry_time=None, alpha=0.95):
        """
        Params:
            events (numpy.array): 0 or 1
            durations (numpy.array): time at which event happened or observation was censored 
            entry_time (numpy.array): time of entry into study (default is None - entry T is 0 for all)
        """
        self.alpha = alpha
        self.label = label
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
        lifetable['se'] = np.sqrt((((risk-failures) * failures)/((risk-1)*risk**2) ).cumsum())
        z = self._compute_z_score()
        lifetable['lower'] = lifetable['cumhaz'] - z * lifetable['se']
        lifetable['upper'] = lifetable['cumhaz'] + z * lifetable['se']
        lifetable['survival'] = np.exp(-lifetable['cumhaz'])
        
        
        self._lifetable = lifetable
    
    def _compute_z_score(self, alpha = None):
        if alpha is None:
            alpha = self.alpha
        return inv_normal_cdf((1. + alpha) / 2.)
    
    def summary(self):
        '''
        Returns the life table
        Time => Nb at risk => Nb of events => Survival => VarSur => CIs => Hazard Rate => Cumlative
        '''
        return self._lifetable
    
    def plot(self, ax):
        
        # Set ax
        c = ax._get_lines.get_next_color()
        self._lifetable['cumhaz'].plot(drawstyle="steps-post", 
                                         c=c, 
                                         label='nelsonaalen_' + str(self.label))
        
        ax.fill_between(self._lifetable.index, 
                        y1=self._lifetable['lower'].values, 
                        y2=self._lifetable['upper'].values, 
                        step='post', 
                        alpha=0.3, 
                        color=c)
        
        return ax