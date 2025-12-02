from typing import List
from userLibrary_helper import OptimizationProblem, PenaltyType

def myFunc(x):
    dimensionality = len(x)

    # ------------------------ PROBLEM-SPECIFIC: Portfolio Data ------------------------ #
    # Mock example of 25 stocks
    Values = [0.12, 0.08, 0.15, 0.09, 0.11, 0.13, 0.07, 0.14, 0.10, 0.09,
              0.16, 0.08, 0.12, 0.11, 0.13, 0.10, 0.09, 0.14, 0.15, 0.12,
              0.08, 0.13, 0.11, 0.07, 0.10]  # Expected returns per unit investment

    Weights = [50, 100, 80, 120, 60, 90, 40, 70, 110, 85,
               95, 55, 65, 75, 105, 60, 80, 90, 100, 50,
               45, 85, 70, 60, 95]  # Price per stock unit

    availableCapital = 1000  # Maximum investment allowed

    preferred_min_stocks = 5   # Soft penalty: minimum diversified stocks
    preferred_max_stocks = 15  # Soft penalty: maximum diversified stocks

    # ------------------------ Define Objective Function ------------------------ #
    def objective(x: List[float], dimensionality: int) -> float:
        objectiveFunctionValue = 0.0
        totalInvestment = 0.0

        # Compute total expected return and total investment
        for i in range(dimensionality):
            objectiveFunctionValue += Values[i] * x[i]
            totalInvestment += Weights[i] * x[i]

        # --- HARD CONSTRAINT ---
        violation = 0.0
        if totalInvestment > availableCapital:
            violation += totalInvestment - availableCapital
        if totalInvestment <= 0:
            violation += abs(totalInvestment)  # penalize empty portfolios

        # --- SOFT CONSTRAINTS ---

        # For Diversification
        num_stocks_invested = sum([1 for xi in x if xi > 0])
        soft_penalty = 0.0
        if num_stocks_invested < preferred_min_stocks:
            soft_penalty += (preferred_min_stocks - num_stocks_invested) * 10.0
        elif num_stocks_invested > preferred_max_stocks:
            soft_penalty += (num_stocks_invested - preferred_max_stocks) * 5.0

        # --- Combine into final fitness ---
        penaltyCoefficients = [100.0]  # weight for hard constraint
        fitness = -objectiveFunctionValue + penaltyCoefficients[0] * violation + soft_penalty

        return fitness

    # ------------------------ Define Constraints List ------------------------ #
    constraints = []  # no external constraints; handled manually in objective

    # ------------------------ Define Penalty Type ------------------------ #
    penalty_type = "No_Penalty"  # Options: No_Penalty, Death_Penalty, Static_Penalty, Debs_Penalty

    if penalty_type == "No_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.No_Penalty)
    elif penalty_type == "Death_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Death_Penalty)
    elif penalty_type == "Static_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Static_Penalty, [100.0])
    elif penalty_type == "Debs_Penalty":
        problem = OptimizationProblem(objective, constraints, PenaltyType.Debs_Penalty)

    # ------------------------ Evaluate Objective ------------------------ #
    return problem.evaluate_objective(x, dimensionality)


# ------------------------ Example Usage ------------------------ #
if __name__ == "__main__":
    # Example decision vector (units of each stock to buy)
    x = [2, 0, 1, 1, 0, 3, 0, 1, 0, 1,
         0, 2, 0, 1, 1, 0, 0, 1, 2, 0,
         0, 1, 1, 0, 1]  # adjust based on desired dimensionality
    fitness = myFunc(x)
    print("Fitness:", fitness)
