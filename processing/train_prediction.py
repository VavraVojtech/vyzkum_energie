import pandas as pd
import numpy as np
import logging
import sys
import os
import types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import datetime
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

### Workaround for Zellij
# Create a dummy MPI communicator that mimics basic functionality.
class DummyMPIComm:
    def Get_rank(self):
        return 0
    def Get_size(self):
        return 1
# Create a dummy MPI module with only the functionality needed.
class DummyMPI:
    COMM_WORLD = DummyMPIComm()
# Inject the dummy mpi4py module into sys.modules.
dummy_mpi4py = types.ModuleType("mpi4py")
dummy_mpi4py.MPI = DummyMPI
sys.modules["mpi4py"] = dummy_mpi4py

# Zellij ---> import is in the function
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# print("MPI initialized. Rank:", comm.Get_rank())
# from zellij.core import Loss, IntVar, FloatVar, CatVar, ArrayVar, ContinuousSearchspace
# from zellij.strategies import Bayesian_optimization

# custom imports
from processing.tools import filter_data, save_model, write_compare_models_to_excel
from config import config
from processing.combiner import get_train_data

def plot_results(df, y_test, y_pred, target, split_date=False):
    results = pd.DataFrame(y_test)
    results = results.rename(columns={target: 'y_test'})
    results['y_pred'] = y_pred
    results = results.merge(df['date'], left_index=True, right_index=True, how='left')

    if split_date:
        results['date'] = [split_date + pd.Timedelta(days=i) for i in range(len(results))]
        results['day_of_week_char'] = results['date'].dt.day_name().str[:2]
    else:
        results.sort_values(by='date', inplace=True, ascending=True)
        results = results.reset_index(drop=True)

    if model_train == 'zellij':
        results.to_csv(f"results/{target}_{model_train}_{strategy}_results.csv", index=False)
    else:
        results.to_csv(f"results/{target}_{model_train}_results.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(15, 11))
    sns.lineplot(data=results, x='date', y='y_test', ax=ax1, color='navy',   linestyle='-', linewidth=3.5, label='y_test')
    sns.lineplot(data=results, x='date', y='y_pred', ax=ax1, color='orange', linestyle=':', linewidth=2.5, label='y_pred')

    upper_bound = results['y_test'] + 0.15*results['y_test']
    lower_bound = results['y_test'] - 0.15*results['y_test']

    ax1.fill_between(results['date'], lower_bound, upper_bound, color='gray', alpha=0.1, label='+- 10 %')
    ax1.set_ylabel(f'{target}', fontsize=14)
    ax1.set_title(f'Graph of validation data of {target}')
    ax1.set_xlabel('Date', fontsize=14)
    
    if len(results) <= 31:
        ax1.set_xticks(results['date'])  # Set the tick locations to each date in 'date' column
        ax1.set_xticklabels([f"{date:%Y-%m-%d} ({day})" for date, day in zip(results['date'], results['day_of_week_char'])], rotation=30, ha="right")

    plt.grid()
    if model_train == 'zellij':
        path = f"graphs/{target}_{model_train}_{strategy}_results_split_date.png" if split_date else f"graphs/{target}_{model_train}_{strategy}_results.png"
    else:
        path = f"graphs/{target}_{model_train}_results_split_date.png" if split_date else f"graphs/{target}_{model_train}_results.png"
    plt.savefig(path)
    print(f"Graph of RESULTS is here: {path}")
    if split_date: print('')

    return plt

def train_test_model(model_type, df, input_cols, target, split_date = False):
    df.columns = df.columns.astype(str)

    if split_date:
        train = df[df['date'] < split_date]         # Training set
        validation = df[df['date'] >= split_date]   # Validation set

        X_all = train[input_cols]
        X_valid = validation[input_cols]
        y_all = train[target]
        y_valid = validation[target]
    else:
        X_all = df[input_cols]
        y_all = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=SET_SEED, test_size=0.2)

    print('')
    logging.info('Size of train set: {}'.format(len(X_train)))
    logging.info('Size of test set: {}'.format(len(X_test)))
    print('')
    logging.info(f'Target is: ---> {y_all.name} <---')
    logging.info(f'Columns used: \n {X_all.columns}')

    model = get_model(X_train, y_train, model_type)         # choose your model
    y_pred = evaluate_model_statistic(model, X_test, y_test)

    if split_date:
        y_test_val = y_valid
        print(f"Statistics since split_date: {split_date.date()}")
        y_pred_val = evaluate_model_statistic(model, X_valid, y_valid)
        return y_test, y_pred, model, y_test_val, y_pred_val
    else:
        return y_test, y_pred, model, None, None

def evaluate_model_statistic(model, X_test, y_test):
    score  = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mean_error = np.mean(y_pred - y_test)
    r2 = r2_score(y_test, y_pred)

    print("")
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('Mean error:', mean_error)
    print('R-squared:', r2)
    print(f"Model Score (on test set): {score}")
    print("")

    # Save results to CSV (append mode)
    results_df = pd.DataFrame([{
        "model": model_type,
        "train": f"{model_train}_{strategy}" if model_train == 'zellij' else model_train,
        "MAE": mae, 
        "RMSE": rmse, 
        "Mean Error": mean_error, 
        "R-squared": r2, 
        "Model Score": score 
    }])
    
    csv_path = f"results/{target}_statistics.csv"
    results_df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)

    return y_pred

def evaluate_model_statistic2(y_test, y_pred):
    print('')
    print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
    # print("RMSE: {}".format(mean_squared_error(y_test, y_pred, squared=False)))   ### squared=False does not work...
    # Manually compute RMSE without squared=False
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: {}".format(rmse))
    print('Mean errror: {}'.format(np.mean(y_pred - y_test)))
    print('R-squared:', r2_score(y_test, y_pred))
    print('')
    # print(f"Model Score (on test set): {score}\n")
    return y_pred

def evaluate_model_statistics3(y_test, y_pred):
    model_type = 'XGBoost'
    model_train = 'hyperopt'
    strategy = "CA"
    target = "spotreba_cr"


    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mean_error = np.mean(y_pred - y_test)
    r2 = r2_score(y_test, y_pred)

    print("")
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('Mean error:', mean_error)
    print('R-squared:', r2)
    print("")

    # Save results to CSV (append mode)
    results_df = pd.DataFrame([{
        "model": model_type,
        "train": f"{model_train}_{strategy}" if model_train == 'zellij' else model_train,
        "MAE": mae, 
        "RMSE": rmse, 
        "Mean Error": mean_error, 
        "R-squared": r2, 
        "Model Score": "NaN" 
    }])
    
    csv_path = f"results/{target}_statistics.csv"
    results_df.to_csv(csv_path, mode='a', header=not pd.io.common.file_exists(csv_path), index=False)

    return y_pred

def get_trained_model(df, model_type, bestparams=None, target='res_coef'):
    df.columns = df.columns.astype(str)
    if target == 'res_coef':
        X_all = df[config.train_columns]
    elif target == 'C_ote':
        X_all = df[config.train_columns_c]
    elif target == 'C_ote_diff_01':
        X_all = df[config.train_columns_C_ote_diff_01]
    elif target == 'C_ote_diff_02':
        X_all = df[config.train_columns_C_ote_diff_02]
    y_all = df[target]

    model = get_model(X_all, y_all, model_type, bestparams)

    return model

def get_model_best_params(model_type, df, input_cols, target, split_date=False, random=False):

    df.columns = df.columns.astype(str)
    scoring = 'r2'
    # 'neg_mean_absolute_error'     : Negative Mean Absolute Error (MAE)
    # 'neg_mean_squared_error'      : Negative Mean Squared Error (MSE)
    # 'neg_root_mean_squared_error' : Negative Root Mean Squared Error (RMSE)
    # 'r2'                          : R-squared (coefficient of determination)
    # 'explained_variance'          : Explained Variance Score
    # 'max_error'                   : Maximum residual error

    if split_date:
        train = df[df['date'] < split_date]  # Training set
        validation = df[df['date'] >= split_date]   # Validation set

        X_all = train[input_cols]
        X_valid = validation[input_cols]

        y_all = train[target]
        y_valid = validation[target]
    else:
        X_all = df[input_cols]
        y_all = df[target]
        # Vojtech: add validation dataset --> it will not draw a graph

    logging.info(f'Target is: ---> {y_all.name} <---')
    logging.info(f'Columns used: \n {X_all.columns}')

    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(random_state=SET_SEED)
        param_grid = {
        'n_estimators':     [300, 400, 500],
        'max_depth':        [None],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1],
        'bootstrap':        [True]
    }

    elif model_type == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(random_state=SET_SEED)
        param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth':    [2, 3, None],
        'learning_rate': [0.1, 0.01, 0.001],
        'loss':         ['squared_error']
    }
        
    elif model_type == 'XGBoost':
        model = XGBRegressor(objective='reg:squarederror', tree_method='hist', random_state=SET_SEED)
        param_grid = {    # play with params
        ################### regular parameters
        'n_estimators':     [100, 200],
        'learning_rate':    [0.001, 0.01, 0.025, 0.05, 0.075, 0.1],
        'reg_alpha':        [0],
        'reg_lambda':       [1],
        ################### control overfitting
        'max_depth':        [3, 4],
        # 'min_child_weight'
        # 'gamma'
        ################### robust to noise
        'subsample':        [0.6, 0.7, 0.8],
        'colsample_bytree': [0.8, 0.9, 1.0],
        # eta
        # num_round
    }

    # create the GridSearchCV object
    if ~random:
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=10,                              # 10-fold cross-validation
            n_jobs=-1,                          # Use all available cores for parallelization
            verbose=1,                          # Optional, to see progress
            scoring=scoring                     # evaluation metric ---> neg_mean_squared_error
        )
    else:
        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,              # Number of parameter combinations to sample (you can adjust)
            cv=10,                  # 10-fold cross-validation
            scoring=scoring,        # Evaluation metric ---> neg_mean_squared_error
            n_jobs=-1,              # Use all available cores for parallelization
            verbose=1,              # Optional, to see progress
            random_state=SET_SEED   # To make results reproducible
        )

    logging.info(f'This might take a while...')
    grid.fit(X_all, y_all)          # fit the GridSearchCV object to the data

    # print the best hyperparameters and the corresponding performance
    logging.info(f'{model_type} parametry: {grid.best_params_}')
    logging.info(f'{model_type} best {scoring} score: {-grid.best_score_}')                         # Negate to get the positive MAE - not sure if this is MAE...
    # logging.info(f'{model_type} avg best score: {(-grid.best_score_)**(1/2)}')              # Negate to get the positive MAE --> this might be closer to MAE: /len(y_all)
    model = grid.best_estimator_

    if split_date:
        y_pred = evaluate_model_statistic(model, X_valid, y_valid)
        return y_valid, y_pred, model, grid.best_params_
    else:
        return None, None, model, grid.best_params_
    
def get_model_best_params_hyperopt(model_type, df, input_cols, target, split_date=False, max_iter=50):
    previous_level = logging.getLogger('hyperopt').level
    logging.getLogger('hyperopt').setLevel(logging.ERROR)

    df.columns = df.columns.astype(str)
    scoring = 'r2'
    # 'neg_mean_absolute_error'     : Negative Mean Absolute Error (MAE)
    # 'neg_mean_squared_error'      : Negative Mean Squared Error (MSE)
    # 'neg_root_mean_squared_error' : Negative Root Mean Squared Error (RMSE)
    # 'r2'                          : R-squared (coefficient of determination)
    # 'explained_variance'          : Explained Variance Score
    # 'max_error'                   : Maximum residual error

    if split_date:
        train = df[df['date'] < split_date]  # Training set
        validation = df[df['date'] >= split_date]   # Validation set
        X_all = train[input_cols]
        X_valid = validation[input_cols]

        y_all = train[target]
        y_valid = validation[target]
    else:
        X_all = df[input_cols]
        y_all = df[target]
        # Vojtech: add validation dataset --> it will not draw a graph

    logging.info(f'Target is: ---> {y_all.name} <---')
    logging.info(f'Columns used: \n {X_all.columns}')
    
    def objective(params):
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(**params, random_state=SET_SEED)
        elif model_type == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(**params, random_state=SET_SEED)
        elif model_type == 'XGBoost':
            model = XGBRegressor(objective='reg:squarederror', tree_method='hist', **params, random_state=SET_SEED)
        else:
            raise ValueError("Unsupported model type")
        
        model.fit(X_all, y_all)
        score = -np.mean(cross_val_score(model, X_all, y_all, scoring=scoring, cv=10, n_jobs=-1))
        return {'loss': score, 'status': STATUS_OK, 'model': model}
    
    param_spaces = {
        'RandomForestRegressor': {
            'n_estimators': hp.choice('n_estimators', list(range(100, 600, 50))),
            'max_depth': hp.choice('max_depth', [None] + list(range(3, 20))),
            'min_samples_split': hp.uniform('min_samples_split', 2, 10),
            'min_samples_leaf': hp.uniform('min_samples_leaf', 1, 5),
            'bootstrap': hp.choice('bootstrap', [True, False]),
        },
        'GradientBoostingRegressor': {
            'n_estimators': hp.choice('n_estimators', list(range(100, 600, 50))),
            'max_depth': hp.choice('max_depth', [None] + list(range(2, 10))),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),
        },
        'XGBoost': {
            'n_estimators': hp.choice('n_estimators', list(range(30, 1000, 10))),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5)),
            'max_depth': hp.randint('max_depth', 3, 12),
            'subsample': hp.uniform('subsample', 0.3, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            # additional ones
            # 'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(10)),
            # 'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(10)),
            # 'gamma': hp.uniform('gamma', 0, 10),
            # 'min_child_weight': hp.loguniform('min_child_weight', np.log(1), np.log(20)),
        }
    }
    
    trials = Trials()
    best_params = fmin(fn=objective, space=param_spaces[model_type], algo=tpe.suggest, max_evals=max_iter, trials=trials, show_progressbar=True)
    best_model = trials.best_trial['result']['model']
    
    logging.getLogger('hyperopt').setLevel(previous_level)
    logging.info(f'{model_type} best parameters: {best_params}')
    
    if split_date:
        y_pred = evaluate_model_statistic(best_model, X_valid, y_valid)
        return y_valid, y_pred, best_model, best_params
    else:
        return None, None, best_model, best_params

########################################
##### Start of Zellij optimization #####
########################################

def get_model_best_params_zellij(model_type, df, input_cols, target='res_coef', split_date=False, strategy = 'GA', max_iter=50):
    # https://zellij.readthedocs.io/en/latest/Welcome/index.html
    # from mpi4py import MPI
    # comm = MPI.COMM_WORLD
    # logging.info("MPI initialized. Rank:", comm.Get_rank())
    from zellij.core import Loss, IntVar, FloatVar, CatVar, ArrayVar, MixedSearchspace, ContinuousSearchspace
    from zellij.utils.neighborhoods import Intervals, ArrayInterval, FloatInterval, IntInterval
    from zellij.strategies.genetic_algorithm import Genetic_algorithm                       # strategy == 'GA':
    from zellij.utils.operators import NeighborMutation, DeapTournament, DeapOnePoint       # strategy == 'GA':
    from zellij.strategies.simulated_annealing import Simulated_annealing                   # strategy == 'SA':
    from zellij.strategies.tools import MulExponential                                      # strategy == 'SA':
    from zellij.strategies import Bayesian_optimization                                     # strategy == 'BO':
    from zellij.strategies.chaos_algorithm import Chaotic_optimization                      # strategy == 'CA':
    from zellij.strategies.tools.chaos_map import Henon, Kent, Logistic, Tent, Random       # strategy == 'CA':

    prev_level = logging.getLogger('zellij').level
    logging.getLogger('zellij').setLevel(logging.ERROR)

    df.columns = df.columns.astype(str)
    scoring = 'r2'
    # 'neg_mean_absolute_error'     : Negative Mean Absolute Error (MAE)
    # 'neg_mean_squared_error'      : Negative Mean Squared Error (MSE)
    # 'neg_root_mean_squared_error' : Negative Root Mean Squared Error (RMSE)
    # 'r2'                          : R-squared (coefficient of determination)
    # 'explained_variance'          : Explained Variance Score
    # 'max_error'                   : Maximum residual error
    
    if split_date:
        train = df[df['date'] < split_date]
        validation = df[df['date'] >= split_date]
        X_all = train[input_cols]
        X_valid = validation[input_cols]
        
        y_all = train[target]
        y_valid = validation[target]
    else:
        X_all = df[input_cols]
        y_all = df[target]
    
    logging.info(f'Target is: ---> {y_all.name} <---')
    logging.info(f'Columns used: \n {X_all.columns}')
    
    if model_type == 'XGBoost':
        if strategy in ['BO', 'CA']:        # required all parameter FloatVar
            n_estimators_var    = FloatVar("n_estimators",      300, 1000)                  # over 1000 could lead to overfitting
            max_depth_var       = FloatVar("max_depth",         3, 10)                      # like this is good
            learning_rate_var   = FloatVar("learning_rate",     0.0001, 0.05)               # like this is good
            subsample_var       = FloatVar("subsample",         0.8, 1.0)                   # randomness for generalization
            colsample_bytree_var= FloatVar("colsample_bytree",  0.8, 1.0)                   # randomness for generalization
            reg_alpha_var       = FloatVar("reg_alpha",         np.log(1e-5), np.log(0.1))  # 
            reg_lambda_var      = FloatVar("reg_lambda",        np.log(1e-5), np.log(1.0))  # 
            gamma_var           = FloatVar("gamma",             0, 0.3)                     # pruning tree
            min_child_weight_var= FloatVar("min_child_weight",  np.log(1), np.log(3))       # like this is good
        elif strategy in ['GA']:
            n_estimators_var    = IntVar("n_estimators",        300, 1000,                  neighbor=IntInterval(50))           # over 1000 could lead to overfitting
            max_depth_var       = IntVar("max_depth",           3, 10,                      neighbor=IntInterval(0.005))        # like this is good
            learning_rate_var   = FloatVar("learning_rate",     0.0001, 0.05,               neighbor=FloatInterval(1))          # like this is good
            subsample_var       = FloatVar("subsample",         0.8, 1.0,                   neighbor=FloatInterval(0.05))       # randomness for generalization
            colsample_bytree_var= FloatVar("colsample_bytree",  0.8, 1.0,                   neighbor=FloatInterval(0.05))       # randomness for generalization
            reg_alpha_var       = FloatVar("reg_alpha",         np.log(1e-5), np.log(0.1),  neighbor=FloatInterval(0.1))        # 
            reg_lambda_var      = FloatVar("reg_lambda",        np.log(1e-5), np.log(1.0),  neighbor=FloatInterval(0.1))        # 
            gamma_var           = FloatVar("gamma",             0, 0.3,                     neighbor=FloatInterval(0.05))       # pruning tree
            min_child_weight_var= FloatVar("min_child_weight", np.log(1), np.log(3),        neighbor=FloatInterval(0.1))        # like this is good
        elif strategy in ['SA']:        # required all parameter FloatVar 
            n_estimators_var    = FloatVar("n_estimators",      300, 1000,                  neighbor=FloatInterval(50))
            learning_rate_var   = FloatVar("learning_rate",     0.0001, 0.05,               neighbor=FloatInterval(0.005))
            max_depth_var       = FloatVar("max_depth",         3, 10,                      neighbor=FloatInterval(1))
            subsample_var       = FloatVar("subsample",         0.8, 1.0,                   neighbor=FloatInterval(0.05))
            colsample_bytree_var= FloatVar("colsample_bytree",  0.8, 1.0,                   neighbor=FloatInterval(0.05))
            reg_alpha_var       = FloatVar("reg_alpha",         np.log(1e-5), np.log(0.1),  neighbor=FloatInterval(0.1))
            reg_lambda_var      = FloatVar("reg_lambda",        np.log(1e-5), np.log(1.0),  neighbor=FloatInterval(0.1))
            gamma_var           = FloatVar("gamma",             0, 0.3,                     neighbor=FloatInterval(0.05))
            min_child_weight_var= FloatVar("min_child_weight",  np.log(1), np.log(3),       neighbor=FloatInterval(0.1))
        
        # Build the search space
        if strategy in ['GA', 'SA']:
            search_vars = ArrayVar(n_estimators_var, learning_rate_var, max_depth_var, subsample_var,
                                   colsample_bytree_var, reg_alpha_var, reg_lambda_var, gamma_var, min_child_weight_var,
                                   label = "Search space", neighbor=ArrayInterval())
        else:
            search_vars = ArrayVar(n_estimators_var, learning_rate_var, max_depth_var, subsample_var,
                                   colsample_bytree_var, reg_alpha_var, reg_lambda_var, gamma_var, min_child_weight_var,
                                   label = "Search space")

    else:
        raise ValueError("Unsupported model type for Zellij optimization in this example.")
    
    # set best so far
    best_loss_so_far  = [np.inf]
    best_model_so_far = [None]
    
    # Define the loss (objective) function.
    @Loss(save=False, verbose=True)
    def objective_function(x):
        params = {}
        # Iterate directly over search_vars.
        for var, val in zip(search_vars, x):
            # For integer parameters, cast to int.
            if var.label in ['n_estimators', 'max_depth']:
                params[var.label] = int(round(val))
            # For parameters defined in log-space, exponentiate.
            elif var.label in ['reg_alpha', 'reg_lambda', 'min_child_weight']:
                params[var.label] = np.exp(val)
            else:
                params[var.label] = val

        model = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                              random_state=SET_SEED, **params)
        model.fit(X_all, y_all)
        loss = -np.mean(cross_val_score(model, X_all, y_all, scoring=scoring, cv=10, n_jobs=-1))

        # If this evaluation is the best so far, save the model.
        if loss < best_loss_so_far[0]:
            best_loss_so_far[0] = loss
            best_model_so_far[0] = model
            folder = f"model/zellij/{target}_{strategy}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            # You can create a filename that includes the loss for reference.
            filename = os.path.join(folder, f"best_model_{datetime.date.today()}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(model, f)
            logging.info(f"New best model saved with loss {loss} to {filename}")

        return loss, model
    
    # ---> SELECT OPTIMIZATION STRATEGY <---
    # Genetic algirothm
    if strategy == 'GA':
        sp = MixedSearchspace(search_vars, objective_function, neighbor=Intervals(),
                              mutation = NeighborMutation(0.5),
                              selection = DeapTournament(3),
                              crossover = DeapOnePoint())
        ga = Genetic_algorithm(sp,
                               max_iter,                                                            # max_item = 1000
                               pop_size=25,                                                         # 10
                               generation=40,                                                       # 1000
                               elitism=0.5,                                                         # 0.5
                            #    filename='',                                                         # initial solutions
                               verbose=True
                               )           
        best_point, best_score = ga.run()
    # Simulated annealing
    elif strategy == 'SA':
        sp = ContinuousSearchspace(search_vars, objective_function, neighbor=Intervals())
        cooling = MulExponential(alpha=0.85, T0=100, Tend=2, peaks=3)
        sa = Simulated_annealing(sp,
                                 f_calls=max_iter,                                                          # max_item = 100
                                 cooling=cooling,
                                 max_iter = 10,
                                 verbose = True)
        initial_point = sp.random_point()
        # initial_value = objective_function(initial_point)
        # logging.info(f"Initial point: {initial_point} with objective value: {initial_value}")
        best_point, best_score = sa.run(initial_point)
    # Bayesian optimization
    elif strategy == 'BO':
        sp = ContinuousSearchspace(search_vars, objective_function)
        bo = Bayesian_optimization(sp,
                                   max_iter,                                                        # 500
                                   verbose = True,
                                   # surrogate =                                                    # botorch.models.gp_regression.SingleTaskGP
                                   # likelihood =                                                   # gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood
                                   # acquisition =                                                  # botorch.acquisition.monte_carlo.qExpectedImprovement,
                                   initial_size = 10,                                               # 10
                                   )
        best_point, best_score = bo.run()
    # Chaotic optimization
    elif strategy == 'CA':
        map_str = 'Henon'               # options: ['Henon', 'Kent', 'Logistic', 'Tent', 'Random']
        ################################
        ### fixing map error - start ###
        ################################
        if map_str == 'Henon':
            from zellij.strategies.tools.chaos_map import Henon as MyMap
        elif map_str == 'Kent':
            from zellij.strategies.tools.chaos_map import Kent as MyMap
        elif map_str == 'Logistic':
            from zellij.strategies.tools.chaos_map import Logistic as MyMap
        elif map_str == 'Tent':
            from zellij.strategies.tools.chaos_map import Tent as MyMap
        elif map_str == 'Random':
            from zellij.strategies.tools.chaos_map import Random as MyMap
        else:
            logging.error(f"Invalid map_str {map_str}. Please choose one of 'Henon', 'Kent', 'Logistic', 'Tent', 'Random'.")
        class MyMapLen(MyMap):
            def __len__(self):
                # Return the number of vectors (or a fixed value, e.g. the 'vectors' attribute)
                return self.vectors
            def __call__(self, vectors, params):
                # When called, simply return this instance (or reinitialize it if needed)
                return self
        ##############################
        ### fixing map error - end ###
        ##############################
        sp = ContinuousSearchspace(search_vars, objective_function)
        chaos_map_instance = MyMapLen(250, sp.size)
        ch = Chaotic_optimization(                  # line 996: "/ len(self.chaos_map)" needs to be commented to work
            sp,
            f_calls=max_iter,             # total number of calls to the loss function
            chaos_map=chaos_map_instance, # pass the callable class instead of an instance, [Henon, Kent, Logistic, Tent, Random]
            exploration_ratio=0.30,       # fraction of calls devoted to exploration
            levels=(32, 6, 2),            # levels for CGS, CLS, and CFS respectively
            polygon=4,                    # vertex number for rotating polygon in local/fine search
            red_rate=0.5,                 # reduction rate for zooming in on the best solution
        )
        best_point, best_score = ch.run()
    else:
        raise ValueError(f"Unsupported strategy type ---> {strategy} < --- for Zellij.") # no more strategies defined

    
    # Convert the best_point list to a dictionary.
    best_params = {}
    for var, val in zip(search_vars, best_point):
        if isinstance(val, list):
            val = val[0]
        if var.label in ['n_estimators', 'max_depth']:
            best_params[var.label] = int(round(val))
        elif var.label in ['reg_alpha', 'reg_lambda', 'min_child_weight']:
            best_params[var.label] = np.exp(val)
        else:
            best_params[var.label] = val

    best_model = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                              random_state=SET_SEED, **best_params)
    best_model.fit(X_all, y_all)
    
    logging.getLogger('zellij').setLevel(prev_level)
    logging.info(f'{model_type} best parameters: {best_params}')
    logging.info(f'{model_type} best score: {best_score}')
    
    if split_date:
        y_pred = evaluate_model_statistic(best_model, X_valid, y_valid)
        return y_valid, y_pred, best_model, best_params
    else:
        return None, None, best_model, best_params



########################################
##### End of Zellij optimization #####
########################################

def get_model(X_train, y_train, model_type, bestparams = None):

    if bestparams is None:
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(n_estimators = 300, bootstrap = True, random_state = SET_SEED,
                                          max_depth = None, min_samples_leaf = 1, min_samples_split = 2)
        elif model_type == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(n_estimators = 500, random_state = SET_SEED, 
                                              max_depth = 2, min_samples_split = 5,
                                              learning_rate = 0.1, loss='squared_error')
        elif model_type == 'XGBoost':
            model = XGBRegressor(colsample_bytree = 1, learning_rate = 0.1, max_depth = 3, n_estimators = 200,       # 0.9, 0.05, 3, 400
                                 reg_alpha = 0, reg_lambda = 1, subsample = 0.9,
                                 objective='reg:squarederror',#  tree_method='hist',
                                 random_state=SET_SEED
                                 )
        else:
            raise ValueError("Unsupported model_type. Choose 'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBoost'.")

    else: # for case of pretrained model parameters
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(**bestparams)
        elif model_type == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(**bestparams)
        elif model_type == 'XGBoost':
            model = XGBRegressor(**bestparams)
        else:
            raise ValueError("Unsupported model_type. Choose 'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBoost'.")

    model.fit(X_train, y_train)

    return model

def feature_importance(model, model_train, input_cols, target):
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("\nThe provided model does not have feature_importances_ attribute.")
        return None
    
    # Get feature importances from the model
    f_importance = model.feature_importances_
    f_names = input_cols

    if len(f_importance) == 0:
        raise ValueError("\nModel has no feature importances available.")
    
    # if len(f_importance) != len(f_names):
    #     raise ValueError(f"Mismatch: {len(f_importance)} importances vs {len(f_names)} features")

    # Create a DataFrame for better handling and sorting
    feature_df = pd.DataFrame({
        'Feature': f_names,
        'Importance': f_importance
    })

    # Sort by importance in descending order
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Visualization')
    plt.gca().invert_yaxis()  # Display most important features at the top
    plt.tight_layout()

    if model_train == 'zellij':
        path = f'graphs/{target}_{model_train}_{strategy}_feat_imp.png'
    else:
        path = f'graphs/{target}_{model_train}_feat_imp.png'
    plt.savefig(path)
    print(f"\nGraph of Feature Importance is here: {path}")
    print('')

    # Print as a table
    # print("\nFeature Importances:")
    # print(feature_df.to_string(index=False, float_format='{:0.4f}'.format))
    return feature_df

def get_short_name(model_type):
    if model_type == 'RandomForestRegressor':
        short = 'RFR'
    elif model_type == 'GradientBoostingRegressor':
        short = 'GBR'
    elif model_type == 'XGBoost':
        short = 'XGB'
    else:
        short = model_type
    return short

def pick_cols(target):
    if target == 'spotreba_cr':
        input_col = config.input_spotreba_cr
    elif target == 'flex_mnozstvi_+':
        input_col = config.input_cena_plus
    elif target == 'flex_obchod_+':
        input_col = config.input_cena_minu
    elif target == 'flex_cena_+':
        input_col = config.input_mnoz_plus
    elif target == 'flex_mnozstvi_-':
        input_col = config.input_mnoz_minu
    elif target == 'flex_obchod_-':
        input_col = config.input_obch_plus
    elif target == 'flex_cena_-':
        input_col = config.input_obch_minu

    return input_col








if __name__ == '__main__':
    # this script is for training models
    # after model is trained, make sure to rename it and put in production
    # in run_short_term_ng.py we use res_coef for calculation
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format='%(asctime)s %(levelname)s | %(message)s')
    logging.info('')

    # data period for model training
    start_date = pd.to_datetime('2016-07-08')     # '2021-07-25 --- '2024-10-31' testing set for September
    end_date   = pd.to_datetime('2024-11-25')     # until '2023-03-31' --- old training dataset; pd.to_datetime('today') - timedelta(days=3)
    split_date = pd.to_datetime('2024-09-01').normalize() # or False

    # which field do you want to check visualy?
    check_list = False
    # check_list = ['direct_radiation', 'cloud_cover', 'wind_speed_10m', 'relative_humidity_2m', 'dew_point_2m', 'wind_gusts_10m', 'app_temp_omt', 'temp_omt']

    show_missing = False            # [12] is normal, show missing cols and columns in training data
    save_data_to_scv = False        # for future experiments, not required
    correlation_matrix = True       # just have a look :)
    show = False                    # show graphs

    model_train = 'grid_search'          # ['no', 'grid_search', 'hyperopt', 'zellij']
    strategy = 'BO'                 # this is for only --> Zellij <-- ['GA', 'SA', 'BO', 'CA']
    random = False                  # only for 'grid_search' ---> to get the same results use False
    SET_SEED = 4565                 # to get the same results over all (or to make different results)
    max_iter = 150                  # res_coef = 1 hour ~~ 4000 iterations | C_ote = 1 hour ~~ 1200 iterations, all depends on numbre of columns

    target = 'spotreba_cr'          # ['spotreba_cr', 
                                    # 'flex_mnozstvi_+', 'flex_obchod_+', 'flex_cena_+',
                                    # 'flex_mnozstvi_-', 'flex_obchod_-', 'flex_cena_-']

    input_cols = pick_cols(target)

    # model_type = 'RandomForestRegressor'       # RFR (+)      https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    # model_type = 'GradientBoostingRegressor'   # GBR (-)      https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    model_type = 'XGBoost'                       # XGBoost      https://xgboost.readthedocs.io/en/stable/index.html
    # model_type RFR and GBR are refused for Zellij, now continue only with XGBoost

    short = get_short_name(model_type)

    df = get_train_data(start_date, end_date)       # CREATE DATASET <----------------------------
    logging.info(f"There are ---> {df.isnull().any(axis=1).sum()} <--- missing rows.")            # check for missing data
    if show_missing:
        nan_rows = df[df.isna().any(axis=1)]
        nan_columns = nan_rows.loc[:, ['date'] + nan_rows.columns[nan_rows.isna().any()].tolist()]
        print(nan_columns)
    
    # print(df.columns)
    # exclude June, July, August
    # df = df[(df['date'].dt.month >= 10) | (df['date'].dt.month <= 4)].reset_index(drop=True)

    # print(df.columns)

    if save_data_to_scv:
        # nan_rows = df[df[['date'] + [target] + input_cols].isna().any(axis=1)]
        # file_name = f'{config.PROJECT_ROOT}/experiments/experimental_dataset.csv'
        # df.to_csv(file_name, index=False) # This is for experimaental purposes
        # logging.info('File write successful: ' + file_name)
        write_compare_models_to_excel(df[['date'] + [target] + input_cols], suffix = 'experimental_dataset')

    if check_list:
        colors = plt.cm.tab10(range(len([target] + check_list)))  # 'tab10' has 10 distinct colors
        plt.figure(figsize=(16, 9))
        for idx, check in enumerate(check_list):
            plt.plot(df['date'], df[[check]], label=check, color=colors[idx])  # Use a color from the palette
        plt.title(f'Check: {check}')
        plt.xlabel('Date')
        plt.ylabel(check)
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('graphs/check.png')
        print('plt saved successfully experiments/check.png')

    if correlation_matrix:
        column_names = [target] + input_cols
        column_index = [df.columns.get_loc(col) for col in column_names]
        correlation_matrix = df.iloc[:, column_index].corr()
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
        plt.title('Correlation Matrix')
        plt.savefig(f'graphs/correlation_{target}.png')
        print(f"\nCorrelation matrix is here: graphs/correlation_{target}.png")
    
    if model_train == 'no':
        logging.info(f'Evaluating model {model_type} with PRESET PARAMETERS started... patience, it may take a few minutes')
        y_test, y_pred, model, y_test_val, y_pred_val = train_test_model(model_type, df, input_cols, target, split_date)
        if split_date:
            plot_results(df, y_test_val, y_pred_val, target=target, split_date = split_date)
        plot_results(df, y_test, y_pred, target=target, split_date = False)
        feature_importance(model, model_train, input_cols, target=target)
        save_model(model, f'_model_{target}_OM_{short}_set')

    elif model_train == 'grid_search':
        logging.info(f'Training model {model_type} with GRID SEARCH started... patience, it may take a few minutes')
        y_valid, y_pred, model, best_params = get_model_best_params(model_type, df, input_cols, target, split_date, random)
        logging.info('Training model finished')
        plot_results(df, y_valid, y_pred, target, split_date = split_date)
        feature_importance(model, model_train, input_cols, target=target)
        save_model(model, f'_model_{target}_OM_{short}_grid_searh')

    elif model_train == 'hyperopt':
        logging.info(f'Training model {model_type} with HYPEROPT started... patience, it may take a few minutes')
        y_valid, y_pred, model, best_params = get_model_best_params_hyperopt(model_type, df, input_cols, target, split_date, max_iter)
        logging.info('Training model finished')
        plot_results(df, y_valid, y_pred, target, split_date = split_date)
        feature_importance(model, model_train, input_cols, target=target)
        save_model(model, f'_model_{target}_OM_{short}_hyperopt')

    elif model_train == 'zellij':
        logging.info(f'Training model {model_type} with ZELLIJ and {strategy} started... patience, it may take a few minutes')
        y_valid, y_pred, model, best_params = get_model_best_params_zellij(model_type, df, input_cols, target, split_date, strategy, max_iter)
        logging.info('Training model finished')
        plot_results(df, y_valid, y_pred, target, split_date = split_date)
        feature_importance(model, model_train, input_cols, target=target)
        save_model(model, f'_model_{target}_OM_{short}_zellij_{strategy}')
    
    if show:
        plt.show()