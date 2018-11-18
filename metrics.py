import numpy as np
from collections import Counter

def p_survival_time(survival_function, p):
    if survival_function.iloc[-1] > p:
        return np.nan
    else:
        return (survival_function <= p).idxmax(0) 
    
def tolist(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x]

def p_survival_times(survival_times, p):
    """
    Expects a Pandas dataframe made up of survival functions
    """
    p = tolist(p)
    for x in p:
        assert (x<1) and (x>0), "Percentile not a float between 0 and 1. Given {}".format(x)
    df = pd.DataFrame(dict((q, survival_times.apply(lambda x: p_survival_time(x, q))) for q in p))
    
    return df#.iloc[0]
    

def _concordance_index(durations, predicted, events):
    '''
    Concordance index is a value between 0 and 1 which measures the concordance
    in the ranking of survival times in the observed data and as predicted by
    the model.
    
    C = 0.5 ==> random
    C = 1   ==> perfect accuracy
    C = 0   ==> perfect anti-accuracy
    
    Arguments:
        durations: (n,) Numpy vector. Expected survival time
        predicted: (n,) Numpy vector. Predicted survival time
        events: (n,) Numpy vector. Binary (0 = censored, 1 = uncensored)
    
    '''
    
    # Sort by the actual durations
    sort_idx =  np.argsort(durations)
    predicted = predicted[sort_idx]
    durations = durations[sort_idx]
    events = events[sort_idx]
    
    def rank(batch, pool):
        '''
        Iterates over each element in the batch and compares to each element in the pool
        Arguments:
            batch: Numpy vector or list of predicted survival times
            pool: list of predicted survival times
        '''
        
        count = Counter({'n_pairs': 0, 'n_correct': 0, 'n_tied': 0})    
        _cache = {} # Cache the results for fast retrieval if predicted values in the batch are identical
        
        for p in batch:
            
            if p in _cache: # If identical to previously encountered values
                count.update(_cache[p])
            
            else:
                p_count = Counter({'n_pairs': 0, 'n_correct': 0, 'n_tied': 0}) # Initialize
                
                for elem in pool:
                    if p == elem: # Ties
                        p_count['n_tied'] += 1
                    elif p > elem: # Correctly ranked the survival times
                        p_count['n_correct'] += 1
                    p_count['n_pairs'] += 1
                
                _cache[p] = p_count # Cache
                count.update(p_count) # Update the counts
                
        return count
    
    pool = []

    splits = np.where(np.diff(durations) != 0)[0] + 1 # Split the arrays into batch based on actual durations
    counter = Counter({'n_pairs' : 0., 'n_correct' : 0., 'n_tied': 0.})
    
    for batch, batch_E in zip(np.split(predicted, splits),
                              np.split(events, splits)):

        successes = batch[batch_E == 1] # Handles the observed failures first
        counter.update(rank(successes, pool))
        pool.extend(successes) # Append to the pool

        censored = batch[batch_E == 0]
        counter.update(rank(censored, pool)) # Don't append the censored events
       
    return (counter['n_correct'] + counter['n_tied'] / 2.) / counter['n_pairs']