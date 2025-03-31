import pandas as pd
import time
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from deap import base, creator, tools, algorithms
import random

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define the objective function
def objective_function(individual, model_type, X_train, y_train, X_test, y_test):
    max_depth, n_estimators, learning_rate = individual

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

    # Train the model and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae,

def objective_function_2(individual, model_type, X_train, y_train, X_test, y_test):
    max_depth, n_estimators, learning_rate = individual

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

    # Train the model and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2, model

# DEAP Optimization
def optimize_with_deap(model_type, X_train, y_train, X_test, y_test):
    # Define the optimization problem
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize MAE
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("max_depth", random.uniform, 1, 10)
    toolbox.register("n_estimators", random.uniform, 10, 200)
    toolbox.register("learning_rate", random.uniform, 0.01, 1.0)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.max_depth, toolbox.n_estimators, toolbox.learning_rate))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function, model_type=model_type,
                     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Population and evolution parameters
    population = toolbox.population(n=20)
    ngen = 50  # Number of generations
    cxpb = 0.5  # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Run the optimization process
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

    # Extract the best individual
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

if __name__ == "__main__":
    # Define the output file
    output_file = "optimized_model_results_deap.csv"

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

    # Iterate over each dataset
    for data_name, data in datasets:
        print(f"Optimizing {data_name} dataset")
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Iterate over each model
        for model_name, model in models.items():
            print(f"\nOptimizing model {model_name} using DEAP...")
            start_time = time.time()  # Track the time taken for optimization

            # DEAP optimization
            best_params_deap = optimize_with_deap(model_name, X_train, y_train, X_test, y_test)
            mae_deap, r2_deap, model_deap = objective_function_2(best_params_deap, model_name, X_train, y_train, X_test, y_test)
            elapsed_time_deap = time.time() - start_time

            results = {
                'model': model_name,
                'library': 'DEAP',
                'data': data_name,
                'mae': mae_deap,
                'r2': r2_deap,
                'elapsed_time': elapsed_time_deap
            }
            results_df = pd.DataFrame([results])
            results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    print(f"Optimization complete. Results saved to {output_file}")