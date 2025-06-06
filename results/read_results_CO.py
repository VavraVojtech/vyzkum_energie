import pandas as pd
import numpy as np
import logging
import sys
import os
import types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load your CSV file
df = pd.read_csv("results/spotreba_cr_statistics.csv")

# Clean the 'train' column
df['train'] = df['train'].str.replace("zellij_CO_", "")

# Metrics to aggregate
metrics = ['MAE', 'RMSE', 'Mean Error', 'R-squared', 'Model Score']

# Compute AVERAGE values by 'train'
avg_df = df.groupby('train')[metrics].mean().reset_index()
avg_df[['MAE', 'RMSE', 'Mean Error']] = avg_df[['MAE', 'RMSE', 'Mean Error']].round(0).astype(int)

# Compute BEST values by 'train'
# Assuming lower MAE, RMSE, Mean Error are better and higher R², Score are better
best_df = df.loc[
    df.groupby('train')[['MAE']].idxmin().values.flatten()
].reset_index(drop=True)
best_df[['MAE', 'RMSE', 'Mean Error']] = best_df[['MAE', 'RMSE', 'Mean Error']].round(0).astype(int)

# To ensure best R² and Score are also best (optional: keep only rows that are Pareto optimal if needed)

# Output to LaTeX
print("%% AVERAGE VALUES")
print(avg_df.to_latex(index=False, float_format="%.4f"))

print("\n%% BEST VALUES")
print(best_df[['train'] + metrics].to_latex(index=False, float_format="%.4f"))
