from typing import List
from userLibrary_helper import OptimizationProblem, PenaltyType

def myFunc(x):
    dimensionality = len(x)

    # ------------------------ PROBLEM-SPECIFIC: Portfolio Data ------------------------ #

    # --- MOCK DATA ---
    # Analogous to values[]
    arrExpectedReturns = [0.12, 0.08, 0.15, 0.09, 0.11, 0.13, 0.07, 0.14, 0.10, 0.09,
                          0.16, 0.08, 0.12, 0.11, 0.13, 0.10, 0.09, 0.14, 0.15, 0.12,
                          0.08, 0.13, 0.11, 0.07, 0.10]  # Expected returns (decimal)
    arrVolatilities = [0.2, 0.25, 0.18, 0.22, 0.19, 0.21, 0.16, 0.23, 0.2, 0.18,
                       0.24, 0.19, 0.22, 0.2, 0.21, 0.18, 0.17, 0.23, 0.25, 0.2,
                       0.16, 0.21, 0.19, 0.15, 0.2]  # Standard deviations (volatility)
    
    # Analogous to weights[]
    arrPrices = [50, 100, 80, 120, 60, 90, 40, 70, 110, 85,
                 95, 55, 65, 75, 105, 60, 80, 90, 100, 50,
                 45, 85, 70, 60, 95]  # Price per stock unit

    # --- ENCODE PROBLEM PREFERENCES ---
    availableCapital = 1000  # Max capital to invest

    preferred_min_stocks = 5   
    preferred_max_stocks = 15 

    lambda_risk = 1.0   # Arbitrary  

    # ------------------------ Define Objective Function ------------------------ #
    def objective(x: List[float], dimensionality: int) -> float:
        expected_return = 0.0
        total_investment = 0.0
        portfolio_variance = 0.0
        num_selected_stocks = 0
        soft_penalty = 0.0
        violation = 0.0

        for i in range(dimensionality):
            expected_return += arrExpectedReturns[i] * x[i]
            total_investment += arrPrices[i] * x[i]
            portfolio_variance += (x[i] ** 2) * (arrVolatilities[i] ** 2)
            if x[i] > 0:
                num_selected_stocks += 1

        objectiveFunctionValue = -expected_return + lambda_risk * portfolio_variance

        # ------------------------ HARD CONSTRAINTS ------------------------ #

        # HC1 - Impossible to invest more than you have
        if total_investment > availableCapital:
            violation += 1e9

        # HC2 - Must invest something
        if total_investment <= 0.0:
            violation += 1e9

        # HC3 - No shorting
        for xi in x:
            if xi < 0:
                violation += 1e9

        # ------------------------ SOFT CONSTRAINTS ------------------------ #

        # SC1 - Penalize Insufficient Diversification
        if num_selected_stocks < preferred_min_stocks:
            soft_penalty += 10.0 * (preferred_min_stocks - num_selected_stocks)
        # SC2 - Penalize Over-Diversification
        elif num_selected_stocks > preferred_max_stocks:
            soft_penalty += 5.0 * (num_selected_stocks - preferred_max_stocks)

        # ------------------------ FINAL FITNESS FUNCTION ------------------ #
        fitness = objectiveFunctionValue + violation + soft_penalty

        return fitness

    # ------------------------ Constraints ------------------------ #
    constraints = []  # Empty since objective() function does it manually

    # ------------------------ Penalty Type ------------------------ #
    penalty_type = "No_Penalty"

    if penalty_type == "No_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.No_Penalty)
    elif penalty_type == "Death_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Death_Penalty)
    elif penalty_type == "Static_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Static_Penalty, [100.0])
    elif penalty_type == "Debs_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Debs_Penalty)

    # ------------------ Evaluate Objective --------------------- #
    return problem.evaluate_objective(x, dimensionality)


# ------------------------ Example Usage ------------------------ #
if __name__ == "__main__":
    
    x = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
         0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
         0, 1, 1, 0, 1] # 25 decision variables. Binary. 

    fitness = myFunc(x)
    print("Fitness:", fitness)
