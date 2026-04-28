import numpy as np
import pandas as pd
import cvxpy as cp
import warnings

def calculate_risk_budgets(buckets):
    """
    Calculates risk budgets b_i based on an array of bucket assignments (1 to 5).
    Matches the formula: b_i = (1 / bucket_i) / sum(1 / bucket_j)
    
    Parameters:
    - buckets (np.array or pd.Series): An array of integers 1 through 5.
    
    Returns:
    - np.array: A normalized risk budget vector 'b' that sums to 1.
    """
    # Ensure inputs are floats for division
    buckets = np.array(buckets, dtype=float)
    
    # Calculate inverse buckets (lower bucket gets higher allocation)
    inverse_buckets = 1.0 / buckets
    
    # Normalize to sum to 1
    b = inverse_buckets / np.sum(inverse_buckets)
    
    return b

def get_deterministic_risk_buckets(volatilities):
    """
    Generates the 1-5 risk buckets for the Deterministic Baseline using volatility.
    """
    # qcut divides the data into 5 equal-sized quantiles 
    # Labels 1 (lowest vol) to 5 (highest vol)
    ranks = pd.Series(volatilities).rank(method='first')
    buckets = pd.qcut(ranks, q=5, labels=[1, 2, 3, 4, 5])
    return buckets.astype(int).values


def optimize_rbo(cov_matrix, b):
    """
    Executes Risk Budgeting Optimization using the strictly convex logarithmic formulation.
    
    Parameters:
    - cov_matrix (np.ndarray): NxN covariance matrix (40x40).
    - b (np.array): Target risk budget vector of length N (40), summing to 1.
    
    Returns:
    - np.array: Optimal portfolio weights.
    """
    n = len(b)
    
    # Define the unnormalized optimization variable
    x = cp.Variable(n, pos=True)
    
    # Objective: Minimize (1/2 * x^T * Sigma * x) - sum(b_i * ln(x_i))
    # We use cp.quad_form for the variance component and cp.sum(cp.multiply(b, cp.log(x))) for the barrier
    portfolio_variance = 0.5 * cp.quad_form(x, cov_matrix)
    log_barrier = b @ cp.log(x) 
    
    objective = cp.Minimize(portfolio_variance - log_barrier)
    
    # We don't need explicit w >= 0 constraints because the log domain handles it.
    # We don't need sum(w) == 1 here because we normalize post-optimization.
    prob = cp.Problem(objective)
    
    for solver in [cp.CLARABEL, cp.SCS]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=solver)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                break
        except Exception:
            continue
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Solver status: {prob.status}. Defaulting to Equal Weighting.")
        return np.ones(n) / n
        
    x_val = x.value
    
    if x_val is None:
        print("Solver returned None. Defaulting to Equal Weighting.")
        return np.ones(n) / n
        
    
    # Normalize the auxiliary vector to get the final portfolio weights summing to 1
    weights = x_val / np.sum(x_val)
    
    return weights
