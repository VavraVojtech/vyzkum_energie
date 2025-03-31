import pandas as pd

# Load and preprocess the DataFrame
df = pd.read_csv('results/spotreba_cr_statistics.csv')
df.drop(columns=['Model Score'], inplace=True)

df['MAE'] = df['MAE'].apply(lambda x: f"{x:.0f}".rstrip('0').rstrip('.'))
df['RMSE'] = df['RMSE'].apply(lambda x: f"{x:.0f}".rstrip('0').rstrip('.'))
df['Mean Error'] = df['Mean Error'].apply(lambda x: f"{x:.0f}".rstrip('0').rstrip('.'))
df['R-squared'] = df['R-squared'].apply(lambda x: f"{x:.4f}".rstrip('0').rstrip('.'))

# Convert to LaTeX
latex_table = df.to_latex(index=False)
print(latex_table)
