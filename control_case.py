import numpy as np

def control_case(mu):
    """
    Represents an equally-weighted portfolio
    """
    n = len(mu)
    
   
    weights = np.ones(n) / n
        
    return weights
