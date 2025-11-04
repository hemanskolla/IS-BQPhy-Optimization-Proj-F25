import random
random.seed(0)
Prices_global = [random.uniform(10.0, 200.0) for _ in range(20)]
totalBudget_global = int(0.5 * sum(Prices_global))

def Evaluator(x):
    dimensionality = len(x)
    try:
        from userLibrary_helper import OptimizationProblem, PenaltyType
    except ImportError:
        # Mock versions for testing
        class OptimizationProblem:
            def __init__(self, objective_fn, constraints, penalty_type, penalty_coefficients=None):
                self.objective_fn = objective_fn
            def evaluate_objective(self, x, dimensionality):
                return self.objective_fn(x, dimensionality)

        class PenaltyType:
            No_Penalty = "No_Penalty"
    import random
    
    # ------------------------ Problem parameters and data ------------------------ #
    # Set random seed for reproducibility
    random.seed(0)
    # Generate random data for stocks (Price, expected return rate, volatility)
    Prices = Prices_global[:dimensionality]
    # Ensure at least some moderate volatility and correlate return with risk for realism
    Returns_rate = []
    Volatilities = []
    for i in range(dimensionality):
        # Random expected return rate (some stocks can have slightly negative or up to ~25% expected return)
        r = random.uniform(-0.05, 0.25)
        # Random volatility between max(r, base) and r+0.3 (ensure volatility is not much lower than return)
        if r < 0:
            vol = random.uniform(0.05, 0.3)  # for negative return, assign a baseline volatility
        else:
            low_vol = r if r > 0.1 else 0.1
            high_vol = r + 0.3
            if high_vol < low_vol:
                high_vol = low_vol
            vol = random.uniform(low_vol, min(high_vol, 0.5))
        Returns_rate.append(r)
        Volatilities.append(vol)
    # Compute Conditional Value at Risk (CVaR) loss fraction for each stock (e.g., 95% worst-case loss)
    CVaR_loss = []
    for i in range(dimensionality):
        # Approximate 95% CVaR for each stock assuming a normal return distribution
        # CVaR_loss represents the fraction of value lost in the worst 5% scenarios.
        loss_frac = 2.06 * Volatilities[i] - Returns_rate[i]   # ~ (VaR at 95% ~1.645*vol, CVaR ~2.06*vol)
        if loss_frac < 0:
            loss_frac = 0.0  # If a stock's expected return is very high relative to volatility, worst-case loss is 0
        CVaR_loss.append(loss_frac)
    # Define available capital (budget) and diversification requirements
    totalBudget = totalBudget_global   # e.g., half of total prices sum as available capital
    min_stocks = 5      # minimum number of stocks to hold for diversification
    max_stocks = 50     # maximum number of stocks to hold (to avoid over-diversification)
    # Risk aversion factor for the fitness function (controls weight of risk term)
    risk_aversion = 0.05  # can be tuned as needed (higher -> more penalty on risk)
    
    def objective(x: list, dimensionality: int) -> float:
        # ------------------------ Objective function ------------------------ #
        # Calculate total expected return and total risk for the portfolio
        expected_return_total = 0.0
        risk_total = 0.0
        for i in range(dimensionality):
            # Expected return (profit) from stock i = Price[i] * Returns_rate[i] * x[i]
            expected_return_total += Prices[i] * Returns_rate[i] * x[i]
            # Risk (CVaR loss) from stock i = Price[i] * CVaR_loss[i] * x[i]
            risk_total += Prices[i] * CVaR_loss[i] * x[i]
        # Fitness function: maximize expected return minus (risk_aversion * risk)
        raw_score = expected_return_total - risk_aversion * risk_total
        # Convert to a minimization objective (since optimizer minimizes fitness): 
        # use negative of raw_score so that minimizing fitness <=> maximizing raw_score
        objectiveFunctionValue = -raw_score
        
        # -------------------- Constraint functions -------------------- #
        # Constraint 1: Budget constraint (total investment cost ≤ available capital)
        total_cost = 0.0
        for i in range(dimensionality):
            total_cost += Prices[i] * x[i]
        ConstraintBudget = totalBudget - total_cost  # should be >= 0 for feasibility
        violation_budget = 0.0
        if ConstraintBudget < 0.0:
            violation_budget = abs(ConstraintBudget)
        
        # Constraint 2: No short selling (all weights must be ≥ 0)
        violation_negative = 0.0
        for i in range(dimensionality):
            if x[i] < 0.0:
                # Accumulate amount of negative investment to penalize
                violation_negative += abs(x[i])
        
        # Constraint 3: Minimum number of stocks in portfolio (diversification lower bound)
        count = 0
        for i in range(dimensionality):
            if x[i] > 0.0:
                count += 1
        ConstraintMin = count - min_stocks   # should be >= 0 (at least min_stocks selected)
        violation_min = 0.0
        if ConstraintMin < 0.0:
            violation_min = abs(ConstraintMin)
        
        # Constraint 4: Maximum number of stocks in portfolio (diversification upper bound)
        ConstraintMax = max_stocks - count   # should be >= 0 (at most max_stocks selected)
        violation_max = 0.0
        if ConstraintMax < 0.0:
            violation_max = abs(ConstraintMax)
        
        # ------------------ Penalty coefficients for the constraints ------------------- #
        penaltyCoefficients = [100.0, 100.0, 100.0, 100.0]  # Penalty weights for each constraint violation
        Total_ConstraintViolation = (penaltyCoefficients[0] * violation_budget +
                                     penaltyCoefficients[1] * violation_negative +
                                     penaltyCoefficients[2] * violation_min +
                                     penaltyCoefficients[3] * violation_max)
        
        # Calculate final fitness as objective value plus total penalty
        fitness = objectiveFunctionValue + Total_ConstraintViolation
        # print(f"Expected Return: {expected_return_total:.2f}")
        # print(f"Risk (CVaR Proxy): {risk_total:.2f}")
        # print(f"Raw Score: {raw_score:.2f}")
        # print(f"Total Cost: {total_cost:.2f}, Budget Left: {ConstraintBudget:.2f}")
        # print(f"Num Stocks Selected: {count}")
        # print(f"Penalty: {Total_ConstraintViolation:.2f}")
        return fitness
    
    # -------------------- Optimization problem instance -------------------- #
    constraints = []  # (Constraints are handled via penalties, so this list remains empty)
    penalty_type = "No_Penalty"  # Using no built-in penalty (we apply penalties manually above)
    
    if penalty_type == "No_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.No_Penalty)
    elif penalty_type == "Death_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Death_Penalty)
    elif penalty_type == "Static_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Static_Penalty)
    elif penalty_type == "Debs_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Debs_Penalty)
    
    # Evaluate and return the fitness for the given decision vector x
    return problem.evaluate_objective(x, dimensionality)


if __name__ == "__main__":
    import numpy as np

    # Number of stocks
    num_stocks = 20

    # Use shared global prices
    from __main__ import Prices_global, totalBudget_global

    # Generate portfolio: random % of budget → then convert to shares
    total_budget = 5000
    weights = np.random.rand(num_stocks)
    weights /= np.sum(weights)
    dollar_allocs = weights * totalBudget_global
    share_allocs = dollar_allocs / np.array(Prices_global)  # x[i] = number of shares

    # Call the evaluator
    fitness_score = Evaluator(share_allocs.tolist())

    # Print result
    print(f"Fitness score for test portfolio: {fitness_score:.4f}")
