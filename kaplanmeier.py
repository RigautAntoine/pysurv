import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .stats import inv_normal_cdf


class KaplanMeier():
    
    def __init__(self, events, durations, alpha=0.95, strata=None):
        
        self.kms = []
        
        if strata is None:
            self.kms.append(KaplanMeierFitter(events, durations, label='', alpha=alpha))
        else:
            stratas = np.unique(strata)
            for s in stratas:
                m = (strata == s)
                self.kms.append(KaplanMeierFitter(events[m], durations[m], label=s, alpha=alpha))
                
    def summary(self):
        return [km.summary() for km in self.kms]
    
    def plot(self):
        
        ax = plt.figure().add_subplot(111)
        
        for km in self.kms:
            km.plot(ax=ax)
            
        
        ax.set_ylim(0, 1)
        ax.set_xlim(0)
        ax.set_xlabel('Timeline')
        
        plt.legend(loc='best')
            

class KaplanMeierFitter():
    """
    Non-parametric estimator of the survival function
    for non- or right-censored data.
    
    TO-DO: Strata, confidence interval
    """
    
    def __init__(self, events, durations, label, alpha=0.95):
        """
        Params:
            events (numpy.array): 0 or 1
            durations (numpy.array): time at which event happened or observation was censored 
        """
        self.label=label
        self._fitted = False
        self.events = events
        self.durations = durations
        self.alpha = alpha
        
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
        self._unique_event_times = unique_event_times
        self._survival = lifetable['survival'].values
        self.fitted = True
    
    def _compute_z_score(self, alpha = None):
        if alpha is None:
            alpha = self.alpha
        return inv_normal_cdf((1. + alpha) / 2.)
    
    def _compute_confidence_bounds(self, alpha = None):
        '''
        Kalbfleisch and Prentice (1980) method
        “exponential” Greenwood formula
        https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
        '''
        
        if alpha is not None:
            self.alpha = alpha
        
        _EPSILON = 1e-5
        # Computation of these should be moved to fitting part. Not gonna change
        stable_survival = np.maximum(self._survival, _EPSILON) # Numerical stability with the log
        #stable_survival = self._survival
        
        deaths = self._lifetable['failures'].values
        ns = self._lifetable['at-risk'].values
        
        var_t = stable_survival**2 * np.cumsum(deaths / (ns * (ns - deaths)))
        var_t_p = np.cumsum(deaths / (ns * (ns - deaths))) / np.log(stable_survival)**2 
        
        z = self._compute_z_score()
        
        c1 = np.log(-np.log(stable_survival)) + z * np.sqrt(var_t_p)
        c2 = np.log(-np.log(stable_survival)) - z * np.sqrt(var_t_p)
        
        confidence = pd.DataFrame()
        confidence['time'] = self._unique_event_times
        confidence['at-risk'] = ns
        confidence['failures'] = deaths
        confidence['survival'] = stable_survival
        confidence['var'] = var_t_p
        confidence['lower'] = np.exp(-np.exp(c1))
        confidence['upper'] = np.exp(-np.exp(c2))
        #confidence = confidence.fillna(0)
        
        return confidence
    
    def summary(self):
        '''
        Returns the life table
        Time => Nb at risk => Nb of events => Survival => VarSur => CIs => Hazard Rate => Cumlative
        '''
        return self._lifetable
    
    def plot(self, ax):
        
        # Set ax
        c = ax._get_lines.get_next_color()
        self._lifetable['survival'].plot(drawstyle="steps-post", 
                                         c=c, 
                                         label='km_estimate_' + str(self.label))
        
        confdf = self._compute_confidence_bounds().set_index('time')[['lower', 'upper']]
        
        ax.fill_between(confdf.index, 
                        y1=confdf['lower'].values, 
                        y2=confdf['upper'].values, 
                        step='post', 
                        alpha=0.3, 
                        color=c)
        
        return ax