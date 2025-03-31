import pandas as pd
import time
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from mealpy.swarm_based.PSO import OriginalPSO
from mealpy import FloatVar

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define the objective function for optimization
def objective_function(solution, model_type, X_train, y_train, X_test, y_test):
    # Define the model parameters and training procedure
    max_depth, n_estimators, learning_rate = solution
    
    # Ensure parameters are within valid ranges
    max_depth = int(max(1, max_depth))
    n_estimators = int(max(10, n_estimators))
    learning_rate = max(0.01, min(learning_rate, 1.0))
    
    # Model selection based on model type
    if model_type == "RandomForest":
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    elif model_type == "XGBoost":
        model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42, verbosity=0)
    else:
        raise ValueError("Unsupported model type")
    
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2, model

# MEALPY Optimization
def optimize_with_mealpy(model_type, X_train, y_train, X_test, y_test):
    def problem_func(solution):
        mae, r2, model = objective_function(solution, model_type, X_train, y_train, X_test, y_test)
        return mae
    
    # Define the problem bounds and objective function
    problem = {
        "obj_func": problem_func,
        "bounds": FloatVar(lb=(1, 10, 0.01), ub=(10, 200, 1.0)),
        "minmax": "min",
    }
    
    # Initialize the PSO model with proper arguments
    model = OriginalPSO(epoch=50, pop_size=20, c1=2.05, c2=2.05, w=0.4) ###
    g_best = model.solve(problem)
    return g_best.solution

if __name__ == '__main__':
    # Define the output file
    output_file = "optimized_model_results.csv"

    # Sample datasets
    datasets = [
        ("Diabetes", load_diabetes()),
        ("California Housing", fetch_california_housing()),
        ("Wine", load_wine())
    ]

    # Define the models
    models = {
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": XGBRegressor
    }

    # Initialize results list
    results = []

    # Iterate over each dataset
    for data_name, data in datasets:
        print(f"Optimizing {data_name} dataset")
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Iterate over each model
        for model_name, model in models.items():
            print(f"\nOptimizing model {model_name} using MEALPY...")
            start_time = time.time() # Track the time taken for optimization
            
            # MEALPY optimization
            best_params_mealpy = optimize_with_mealpy(model_name, X_train, y_train, X_test, y_test)
            mae_mealpy, r2_mealpy, model_mealpy = objective_function(best_params_mealpy, model_name, X_train, y_train, X_test, y_test)
            elapsed_time_mealpy = time.time() - start_time

            results = {
                'model': model_name,
                'library': 'MEALPY',
                'data': data_name,
                'mae': mae_mealpy,
                'r2': r2_mealpy,
                'elapsed_time': elapsed_time_mealpy
            }
            results_df = pd.DataFrame([results])
            results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    print(f"It is done. Results saved to {output_file}")