import pandas as pd
import time
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import pygmo as pg

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



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

# Define the objective function for PyGMO
class ModelOptimizationProblem:
    def __init__(self, model_type, X_train, y_train, X_test, y_test):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fitness(self, x):
        max_depth, n_estimators, learning_rate = x
        max_depth = int(max(1, max_depth))
        n_estimators = int(max(10, n_estimators))
        learning_rate = max(0.01, min(learning_rate, 1.0))

        # Select the model based on the type
        if self.model_type == "RandomForest":
            model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        elif self.model_type == "GradientBoosting":
            model = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        elif self.model_type == "XGBoost":
            model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42, verbosity=0)
        else:
            raise ValueError("Unsupported model type")
        
        # Train the model and calculate MAE as the objective
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        return [mae]

    def get_bounds(self):
        return ([1, 10, 0.01], [10, 200, 1.0])  # (lower bounds, upper bounds)

    def get_nobj(self):
        return 1  # Single-objective optimization

# PyGMO Optimization
def optimize_with_pygmo(model_type, X_train, y_train, X_test, y_test):
    problem = pg.problem(ModelOptimizationProblem(model_type, X_train, y_train, X_test, y_test))
    algorithm = pg.algorithm(pg.de(gen=50))  # Differential Evolution
    population = pg.population(problem, size=20)  # Population size
    population = algorithm.evolve(population)
    best_solution = population.champion_x  # Best parameters found
    return best_solution

if __name__ == '__main__':
    # Define the output file
    output_file = "optimized_model_results_pygmo.csv"

    # Sample datasets
    datasets = [
        # ("Diabetes", load_diabetes()),
        ("Wine", load_wine()),
        ("California Housing", fetch_california_housing())    
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
            print(f"\nOptimizing model {model_name} using PyGMO...")
            start_time = time.time()  # Track the time taken for optimization
            
            # PyGMO optimization
            best_params_pygmo = optimize_with_pygmo(model_name, X_train, y_train, X_test, y_test)
            mae_pygmo, r2_pygmo, model_pygmo = objective_function(best_params_pygmo, model_name, X_train, y_train, X_test, y_test)
            elapsed_time_pygmo = time.time() - start_time

            results = {
                'model': model_name,
                'library': 'PyGMO',
                'data': data_name,
                'mae': mae_pygmo,
                'r2': r2_pygmo,
                'elapsed_time': elapsed_time_pygmo
            }
            results_df = pd.DataFrame([results])
            results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    print(f"Optimization complete. Results saved to {output_file}")