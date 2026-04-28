import numpy as np
import cvxpy as cp
import warnings

def optimize_mvo(mu, cov_matrix, risk_aversion=1.0):
    """
    Executes a single-period (annual) Mean-Variance Optimization.
    
    Parameters:
    - mu (np.array): Expected returns vector of length N (40).
    - cov_matrix (np.ndarray): NxN covariance matrix (40x40).
    - risk_aversion (float): The lambda penalty parameter scaling risk tolerance.
    
    Returns:
    - np.array: Optimal portfolio weights.
    """
    n = len(mu)
    
    # Set portfolio weights
    w = cp.Variable(n)
    
    # Expected Return: w^T * mu
    expected_return = w @ mu
    
    # Ensure PSD on covariance matrix
    cov_matrix = cov_matrix + 1e-8 * np.eye(n)
    
    # Portfolio Variance: w^T * Sigma * w
    portfolio_variance = cp.quad_form(w, cov_matrix)
    
    # Objective: Maximize return penalized by risk
    objective = cp.Maximize(expected_return - risk_aversion * portfolio_variance)
    
    # Constraints: Fully invested (sum to 1) and Long-only (w >= 0)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 0.1
    ]
    
    # Formulate and solve the convex optimization problem
    prob = cp.Problem(objective, constraints)
    
    for solver in [cp.OSQP, cp.CLARABEL, cp.SCS]:
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
        

    # Extract the optimal weights
    weights = w.value
    
    # Fallback if the solver returned None (e.g., infeasible due to bad data inputs)
    if weights is None:
        print("Solver returned None. Defaulting to Equal Weighting.")
        return np.ones(n) / n
        
    # Clean up numerical noise
    weights[weights < 1e-5] = 0.0
    
    # Re-normalize to ensure the weights sum to exactly 1.0 after cleaning
    weights = weights / np.sum(weights)
    
    return weights
