
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, fetch_california_housing, load_wine
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import optuna
from hyperopt import fmin, tpe, hp, Trials
from skopt import BayesSearchCV
from skopt.space import Real, Integer

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_param_grid(model, library="optuna"):
    """
    Creates a parameter grid based on the model type and the optimization library.
    """
    if isinstance(model, RandomForestRegressor):
        # RandomForestRegressor specific parameters
        if library == "optuna":
            return {
                "n_estimators": (50, 200),
                "max_depth": (3, 12),
            }
        elif library == "hyperopt":
            return {
                "n_estimators": hp.quniform("n_estimators", 50, 200, 1),
                "max_depth": hp.quniform("max_depth", 3, 12, 1),
            }
        elif library == "skopt":
            return {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(3, 12),
            }
        
    elif isinstance(model, GradientBoostingRegressor):
        # GradientBoostingRegressor specific parameters
        if library == "optuna":
            return {
                "n_estimators": (50, 200),  # will use trial.suggest_int()
                "max_depth": (3, 12),  # will use trial.suggest_int()
                "learning_rate": (0.01, 0.1),  # will use trial.suggest_uniform()
            }
        elif library == "hyperopt":
            return {
                "n_estimators": hp.quniform("n_estimators", 50, 200, 1),
                "max_depth": hp.quniform("max_depth", 3, 12, 1),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
            }
        elif library == "skopt":
            return {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(3, 12),
                "learning_rate": Real(0.01, 0.1),
            }

    elif isinstance(model, xgb.XGBRegressor):
        # XGBRegressor specific parameters
        if library == "optuna":
            return {
                "n_estimators": (50, 200),  # will use trial.suggest_int()
                "max_depth": (3, 12),  # will use trial.suggest_int()
                "learning_rate": (0.01, 0.1),  # will use trial.suggest_uniform()
                "subsample": (0.5, 1.0),  # will use trial.suggest_uniform()
                "colsample_bytree": (0.5, 1.0),  # will use trial.suggest_uniform()
            }
        elif library == "hyperopt":
            return {
                "n_estimators": hp.quniform("n_estimators", 50, 200, 1),
                "max_depth": hp.quniform("max_depth", 3, 12, 1),
                "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            }
        elif library == "skopt":
            return {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(3, 12),
                "learning_rate": Real(0.01, 0.1),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
            }

    return {}

def optuna_optimization(model, X_train, y_train, output_file, name_data, library="optuna"):
    """
    Performs optimization using Optuna to find the best parameters for a given model.
    Saves the results in a CSV file.
    """
    start_time_all = time.time()

    # Create the parameter grid dynamically based on the model and library
    param_grid = create_param_grid(model, library)

    # Define the objective function
    def objective(trial):
        # Sample hyperparameters from the grid
        params = {key: trial.suggest_int(key, *value) if isinstance(value, tuple) else trial.suggest_float(key, *value) 
                  for key, value in param_grid.items()}

        # Set model parameters
        model.set_params(**params)

        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_train)

        # Calculate MAE and R^2 score
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        return mae  # Return the loss (objective function value)

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='minimize')  # Minimize MAE
    study.optimize(objective, n_trials=300) # Vojtech

    # Get the best hyperparameters and model performance
    best_params = study.best_params
    best_value = study.best_value

    print(f"Best parameters found: {best_params}")
    print(f"Best MAE: {best_value}")

    # Set the model with the best parameters and retrain
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Make predictions with the best model
    y_pred = model.predict(X_train)

    # Calculate MAE, R², and elapsed time
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    elapsed_time = time.time() - start_time_all

    print(f"MAE: {mae}")
    print(f"R^2: {r2}")
    print(f"Elapsed Time: {elapsed_time} seconds")

    # Prepare the results for saving
    results = {
        'model': str(model.__class__.__name__),
        'library': 'optuna',
        'data': name_data,
        'mae': mae,
        'r2': r2,
        'elapsed_time': elapsed_time
    }

    # Convert to DataFrame and append to the output CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    return mae, r2, elapsed_time

def hyperopt_optimization(model, X_train, y_train, output_file, name_data, library="hyperopt"):
    """
    Performs optimization using Hyperopt to find the best parameters for a given model.
    Saves the results in a CSV file.
    """
    start_time_all = time.time()

    # Create the parameter grid dynamically based on the model and library
    param_grid = create_param_grid(model, library)

    def objective(params):
        # Ensure integer hyperparameters are cast as integers
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])  # Cast max_depth to int

        # Set model parameters
        model.set_params(**params)

        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_train)

        # Calculate MAE and R^2 score
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        return {'loss': mae, 'status': 'ok'}  # Return loss for Hyperopt optimization

    # Initialize trials and run the optimization
    trials = Trials()
    best = fmin(fn=objective, space=param_grid, algo=tpe.suggest, max_evals=300, trials=trials) # Vojtech
    

    best_params = best
    print("Best parameters found:", best_params)

    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])  # Cast max_depth to int

    # Refit the model with best parameters
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Make predictions again with the best model
    y_pred = model.predict(X_train)

    # Calculate MAE, R^2, and elapsed time
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    elapsed_time = time.time() - start_time_all

    print(f"MAE: {mae}")
    print(f"R^2: {r2}")
    print(f"Elapsed Time: {elapsed_time} seconds")

    # Prepare the results for saving
    results = {
        'model': str(model.__class__.__name__),
        'library': 'hyperopt',
        'data': name_data,
        'mae': mae,
        'r2': r2,
        'elapsed_time': elapsed_time
    }

    # Convert to DataFrame and append to the output CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    return mae, r2, elapsed_time

def skopt_optimization(model, X_train, y_train, output_file, name_data, library="skopt"):
    """
    Performs optimization using Skopt to find the best parameters for a given model.
    Saves the results in a CSV file.
    """
    start_time_all = time.time()

    # Create the parameter grid dynamically based on the model and library
    param_grid = create_param_grid(model, library)

    # Define the Skopt BayesSearchCV
    opt = BayesSearchCV(model, param_grid, n_iter=300, cv=3, n_jobs=-1, verbose=0) # Vojtech

    # Fit the model
    opt.fit(X_train, y_train)

    # Get the best parameters and model performance
    best_params = opt.best_params_
    best_value = opt.best_score_

    print(f"Best parameters found: {best_params}")
    print(f"Best score (negative MAE): {best_value}")

    # Set the model with the best parameters and retrain
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Make predictions with the best model
    y_pred = model.predict(X_train)

    # Calculate MAE, R², and elapsed time
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    elapsed_time = time.time() - start_time_all

    print(f"MAE: {mae}")
    print(f"R^2: {r2}")
    print(f"Elapsed Time: {elapsed_time} seconds")

    # Prepare the results for saving
    results = {
        'model': str(model.__class__.__name__),
        'library': 'skopt',
        'data': name_data,
        'mae': mae,
        'r2': r2,
        'elapsed_time': elapsed_time
    }

    # Convert to DataFrame and append to the output CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    return mae, r2, elapsed_time

def choose_optimization_library(library_name, model, X_train, y_train, name_data):
    """
    Choose the optimization library based on user preference (Optuna, Hyperopt, or Skopt).
    """
    output_file = f'experiment_ML_MH.csv'

    if library_name == 'optuna':
        return optuna_optimization(model, X_train, y_train, output_file, name_data, library=library_name)
    elif library_name == 'hyperopt':
        return hyperopt_optimization(model, X_train, y_train, output_file, name_data, library=library_name)
    elif library_name == 'skopt':
        return skopt_optimization(model, X_train, y_train, output_file, name_data, library=library_name)
    else:
        raise ValueError("Optimization library not recognized.")

if __name__ == '__main__':

    # Define the datasets
    datasets = [
        ("Diabetes", load_diabetes),
        ("California Housing", fetch_california_housing),
        ("Wine", load_wine)
    ]

    optimization_libraries = ['optuna', 'hyperopt', 'skopt']

    for optimization_library in optimization_libraries:
        for name_data, dataset in datasets:
            print(f"Optimizing {name_data} dataset")

            # Load dataset
            data = dataset()
            X = pd.DataFrame(data.data)
            y = pd.Series(data.target)

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define models
            models = [
                RandomForestRegressor(),
                GradientBoostingRegressor(),
                xgb.XGBRegressor()
            ]

            # Perform optimization using the selected library
            for model in models:
                print(f"\nOptimizing model {model.__class__.__name__} using {optimization_library}...")
                choose_optimization_library(optimization_library, model, X_train, y_train, name_data)