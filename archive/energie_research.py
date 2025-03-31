import pandas as pd
import time
import numpy as np

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from experiment.experiment_PyGMO import optimize_with_pygmo, objective_function

if __name__ == '__main__':
    output_file = "energy_results.csv"
    data = pd.read_csv('input_data/OTE_NG_REPORT.csv')
    data.rename(columns={'DATE': 'date'}, inplace=True)
    data_weather = pd.read_csv('input_data/OpenMeteo_train_data_0_24.csv')

    data = data.merge(data_weather, on='date', how='left')

    target = 'flex_cena_+'
    features = ['temperature_2m','relative_humidity_2m','dew_point_2m',
                'apparent_temperature','cloud_cover','wind_speed_10m',
                'wind_gusts_10m','direct_radiation']

    print(data)

    X = data[target]
    y = data[features]
    # multicolinearity check
    corr_matrix = data[[target] + features].corr()
    print(corr_matrix)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {"XGBoost": XGBRegressor}
    
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
            'data': 'ng',
            'mae': mae_pygmo,
            'r2': r2_pygmo,
            'elapsed_time': elapsed_time_pygmo
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)

    print(f"Optimization complete. Results saved to {output_file}")