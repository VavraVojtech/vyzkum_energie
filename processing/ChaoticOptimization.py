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

from processing.train_prediction import pick_cols, get_short_name, get_model_best_params_zellij, plot_results, feature_importance, save_model
from processing.combiner import get_train_data
from processing.tools import save_model

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
    split_date = pd.to_datetime('2023-11-25').normalize() # or False

    show_missing = False            # [12] is normal, show missing cols and columns in training data
    show = False                    # show graphs

    df = get_train_data(start_date, end_date)       # CREATE DATASET <----------------------------
    logging.info(f"There are ---> {df.isnull().any(axis=1).sum()} <--- missing rows.")            # check for missing data
    if show_missing:
        nan_rows = df[df.isna().any(axis=1)]
        nan_columns = nan_rows.loc[:, ['date'] + nan_rows.columns[nan_rows.isna().any()].tolist()]
        print(nan_columns)

    # model_type = 'RandomForestRegressor'       # RFR (+)      https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    # model_type = 'GradientBoostingRegressor'   # GBR (-)      https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    model_type = 'XGBoost'                       # XGBoost      https://xgboost.readthedocs.io/en/stable/index.html
    # model_type RFR and GBR are refused for Zellij, now continue only with XGBoost
    short = get_short_name(model_type)


    model_train = 'zellij'          # ['no', 'grid_search', 'hyperopt', 'zellij']
    random = False                  # only for 'grid_search' ---> to get the same results use False
    strategy = 'CO'                 # this is for only --> Zellij <-- ['GA', 'SA', 'BO', 'CO']
    max_iter = 100                  # res_coef = 1 hour ~~ 4000 iterations | C_ote = 1 hour ~~ 1200 iterations, all depends on numbre of columns

    target = 'spotreba_cr'          # ['spotreba_cr', 
                                    # 'flex_mnozstvi_+', 'flex_obchod_+', 'flex_cena_+',
                                    # 'flex_mnozstvi_-', 'flex_obchod_-', 'flex_cena_-']
    input_cols = pick_cols(target)


    map_options = ['Random'] # 'Henon', 'Kent', 'Logistic', 'Tent', 
    set_seed_list = [52, 673, 3874, 15675, 423696, 4365787, 45638798, 556347929, 1235867040] # 1

    for map_str in map_options:
        for SET_SEED in set_seed_list:
            logging.info('')
            logging.info(f"map: {map_str} + SET_SEED: {SET_SEED}")
            logging.info(f'Training model {model_type} with ZELLIJ and {strategy} and {map_str} started... patience, it may take a few minutes')
            y_valid, y_pred, model, best_params = get_model_best_params_zellij(model_type, df, input_cols, target, split_date, model_train, strategy, map_str, max_iter, SET_SEED)
            logging.info('Training model finished')
            plot_results(df, y_valid, y_pred, target, model_train, strategy, map_str, split_date, SET_SEED)
            feature_importance(model, model_train, strategy, map_str, input_cols, target)
            if strategy == 'CO':
                save_model(model, f'_model_{target}_OM_{short}_zellij_{strategy}_{map_str}')
            else:
                save_model(model, f'_model_{target}_OM_{short}_zellij_{strategy}')