import pandas as pd

# Load and preprocess the DataFrame
df = pd.read_csv('experiment_ML_MH.csv')
df = df[['data', 'library', 'model', 'mae', 'r2', 'elapsed_time']]
df.rename(columns={
    'data'   : 'Data',
    'library': 'MH library',
    'model'  : 'ML model',
    'mae'    : 'MAE',
    'r2'     : 'R^2',
    'elapsed_time': 'Time (sec)'},
    inplace=True)

df.replace({
    'RandomForestRegressor': 'RFR',
    'RandomForest': 'RFR',
    'GradientBoostingRegressor': 'GBR',
    'GradientBoosting' : 'GBR',
    'XGBRegressor': 'XGBoost',
    'hyperopt': 'Hyperopt',
    'optuna': 'Optuna',
    'skopt': 'Skopt'},
    inplace=True)

# Round values and format
df['MAE'] = df['MAE'].apply(lambda x: f"{x:.4f}".rstrip('0').rstrip('.'))
df['R^2'] = df['R^2'].apply(lambda x: f"{x:.4f}".rstrip('0').rstrip('.'))
df['Time (sec)'] = round(df['Time (sec)'], 0).astype(int)

# Sort the DataFrame
df.sort_values(by=['Data', 'MH library', 'ML model'], inplace=True)

df = df[df['Data'] == 'California Housing'][['MH library', 'ML model', 'MAE', 'R^2', 'Time (sec)']]

# Convert to LaTeX
latex_table = df.to_latex(index=False)
print(latex_table)
