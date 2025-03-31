
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
import joblib


from processing.train_prediction import evaluate_model_statistic2, evaluate_model_statistics3



if __name__ == '__main__':

    model_type = 'XGBoost'
    model_train = 'hyperopt'
    strategy = "CA"

    if model_train == 'zellij':
        df = pd.read_csv(f'results/spotreba_cr_{model_train}_{strategy}_results.csv')
    else:
        df = pd.read_csv(f'results/spotreba_cr_{model_train}_results.csv')
    y_test = df['y_test']
    y_pred = df['y_pred']
    evaluate_model_statistics3(y_test, y_pred)
