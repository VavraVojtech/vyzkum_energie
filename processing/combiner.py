import logging
import sys
import holidays
import pandas as pd
import numpy as np
from datetime import timedelta

import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processing.tools import filter_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def get_train_data(start_date, end_date, timeseries_graph=False, corr_matrix=False):

    # add spotreba_cr
    data_odch = pd.read_csv('input_data/OTE_NG_ODCHYLKY_NC_BAL.csv')
    data_odch = data_odch[['DATE', 'spotreba_cr']]
    data_odch['spotreba_cr'] = data_odch['spotreba_cr'] * (-1)

    # add targets
    data = pd.read_csv('input_data/OTE_NG_REPORT.csv')
    data = data.merge(data_odch, on='DATE', how='left')
    data.rename(columns={'DATE': 'date'}, inplace=True)

    # add targets lags
    list_of_targets = ['flex_mnozstvi_+', 'flex_obchod_+', 'flex_cena_+',
                       'flex_mnozstvi_-', 'flex_obchod_-', 'flex_cena_-',
                       'spotreba_cr']
    for target in list_of_targets:
        data = shift_variable(data, target)
        # add targets rolls
        for i in range(2,5):
            data = get_rolling_stat(data, target, num=i)

    # add weather
    data_weather = pd.read_csv('input_data/OpenMeteo_train_data_6_6.csv')
    data_weather.drop(columns=['visibility'], inplace=True)
    data = data.merge(data_weather, on='date', how='left')

    # add lags and rolls
    data = shift_temperatures(data, 'temperature_2m')
    data = shift_temperatures(data, 'apparent_temperature')
    data = shift_variable(data, 'direct_radiation')
    data = shift_variable(data, 'cloud_cover')
    data = shift_variable(data, 'wind_speed_10m')
    data = shift_variable(data, 'relative_humidity_2m')
    data = shift_variable(data, 'dew_point_2m')
    data = shift_variable(data, 'wind_gusts_10m')
    for i in range(2,5):
        data = get_rolling_stat(data, 'temperature_2m', num=i)

    # add spcecial variables
    data = get_omt_diffs(data)
    data = determine_heating_period(data, start_date=None, end_date=None, temp='temperature_2m', suffix='_OM')
    data = compute_no_daydegrees(data, temp='temperature_2m', date_col='date', dd_col='dd_OM')

    # add date info
    data = set_date_features(data)       # + add data
    
    # filter data and 
    data = filter_data(data, start_date, end_date)
    data['date'] = pd.to_datetime(data['date'])
    # data = data.set_index('date')

    if corr_matrix:
        corr_matrix = data.drop(columns=['date']).corr() # multicolinearity check
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Korelační matice")
        plt.savefig("graphs/correlation.png", dpi=300, bbox_inches='tight')
        print('Graph CORELATIONdone: graphs/correlation.png')

    # timeseries graf
    if timeseries_graph:
        fig, axes = plt.subplots(nrows=len(data.columns), ncols=1, figsize=(12, 3 * len(data.columns)))
        for ax, col in zip(axes, data.columns):
            ax.plot(data['date'], data[col], label=col)
            ax.set_title(col)
            ax.set_xlabel("Date")
            ax.legend()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig("graphs/time_series_plots.png", dpi=300, bbox_inches='tight')
        print('Graph TIMESERIES_GRAPH done: graphs/time_series_plots.png')

    return data

def get_omt_diffs(df_OM):

    df_OM['mix_temperature_diff_00'] = df_OM['temperature_2m'] - df_OM['apparent_temperature']
    df_OM['mix_temperature_diff_11'] = df_OM['temperature_2m_1d'] - df_OM['apparent_temperature_1d']
    df_OM['mix_temperature_diff_22'] = df_OM['temperature_2m_2d'] - df_OM['apparent_temperature_2d']

    df_OM['mix_temperature_diff_00_11'] = df_OM['mix_temperature_diff_00'] - df_OM['mix_temperature_diff_11']
    df_OM['mix_temperature_diff_00_22'] = df_OM['mix_temperature_diff_00'] - df_OM['mix_temperature_diff_22'] # this is to calculate change in feeling temperature

    for i in [1, 2]:
        df_OM = make_difference(df_OM, 'temperature_2m',       i)
        df_OM = make_difference(df_OM, 'apparent_temperature', i)
        df_OM = make_difference(df_OM, 'relative_humidity_2m', i)
        df_OM = make_difference(df_OM, 'dew_point_2m',         i)
        df_OM = make_difference(df_OM, 'cloud_cover',          i)
        df_OM = make_difference(df_OM, 'wind_speed_10m',       i)
        df_OM = make_difference(df_OM, 'wind_gusts_10m',       i)

    weather_list = ['temperature_2m', 'apparent_temperature', 'direct_radiation', 'relative_humidity_2m', 'dew_point_2m',
                    'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m']
    
    for feature in weather_list:
        df_OM[f"{feature}_diff_01"] = df_OM[feature] - df_OM[f"{feature}_1d"]

    for feature in weather_list:
        df_OM[f"{feature}_diff_02"] = df_OM[feature] - df_OM[f"{feature}_2d"]

    return df_OM

def make_difference(df, variable, num):
    df[f'{variable}_diff_0{num}'] = df[variable] - df[variable].shift(num)
    return df

def get_rolling_stat(df, variable, num = 7):
    df[f'roll_{variable}_mean_{num}'] = df[variable].rolling(window=num).mean().shift(1)
    # df[f'roll_{variable}_std_{num}'] = df[variable].rolling(window=num).std().shift(1)
    return df

def set_date_features(df):
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day_of_week'] = pd.DatetimeIndex(df['date']).dayofweek +1
    cz_holidays = holidays.Czechia()

    # not (holiday or weekend)
    df['is_weekend'] = pd.DatetimeIndex(df['date']).dayofweek >= 5
    df['is_holiday'] = df['date'].apply(lambda x: x in cz_holidays)
    df['is_working_day'] = ~(df['is_weekend'] | df['is_holiday'])

    # day of year for seasonality
    df['day_of_year'] = pd.DatetimeIndex(df['date']).dayofyear
    df['seasonality'] = df['day_of_year'].apply(estimate_night_length)
    df['season'] = df['month'].apply(get_season)
    df = df.drop(columns=['day_of_year'])

    # Ensure the 'date' column is datetime type
    df['date'] = pd.to_datetime(df['date'])
    df['water_year_start'] = df['date'].apply(
        lambda d: pd.Timestamp(year=d.year if d.month >= 9 else d.year - 1, month=9, day=1)
    )
    df['posan'] = (df['date'] - df['water_year_start']).dt.days + 1
    
    df.drop(columns='water_year_start', inplace=True)

    return df

def get_season(month):
        if month in [12, 1, 2]:
            return 10 # 'Winter'
        elif month in [3, 4, 5]:
            return 4 # 'Spring'
        elif month in [6, 7, 8]:
            return 1 # 'Summer'
        elif month in [9, 10, 11]:
            return 4 #'Autumn'
        else:
            print('WARNING! There is no such month!')
            return None

def estimate_night_length(day_of_year):
    # Approximate the longest night around Dec 21 (winter solstice, ~355th day)
    # and the shortest night around June 21 (summer solstice, ~172nd day).
    # Using a sinusoidal approximation with a period of 365.25 days.
    return 0.5 * (1 - np.cos(2 * np.pi * (day_of_year - 172) / 365.25))

def determine_heating_period(df, start_date=None, end_date=None, temp='temp', suffix=''):
    # https://www.chmi.cz/historicka-data/pocasi/otopna-sezona
    df['date'] = pd.to_datetime(df['date'])
    
    if start_date is not None:
        df = filter_data(df, start_date - timedelta(days=4), end_date)
    df.reset_index(drop=True, inplace=True)

    # Initialize heating column and penalization factor
    df[f'heating{suffix}'] = float(0)
    penalization = 1

    # Define summer and heating period masks
    summer_mask = df['date'].dt.month.between(6, 8)
    heating_period_mask = df['date'].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5])

    # Check if heating is active at the start
    heating_active = not summer_mask.iloc[0]

    for i in range(1, len(df)):
        current_temp = df.loc[i, temp]
        temp_conditions = (
            df.loc[i, f'{temp}_1d'] < 13 and
            df.loc[i, f'{temp}_2d'] < 13 and
            current_temp < 13 and
            df.loc[i, f'{temp}_f_1d'] < 13
        )
        
        # Handle summer period
        if summer_mask.iloc[i]:
            heating_active = False
            df.loc[i, f'heating{suffix}'] = float(0)
        
        # CHMU definition of heating period
        elif heating_period_mask.iloc[i] and temp_conditions:
            heating_active = True
            penalization = 1
            df.loc[i, f'heating{suffix}'] = float(1)

        # Penalize heating when temperature is >= 13°C
        elif heating_active and current_temp >= 13:
            penalization *= 0.3
            df.loc[i, f'heating{suffix}'] = float(penalization)
        
        # Activate heating when temperature < 13°C in heating period
        elif heating_active and current_temp < 13:
            penalization = 1
            df.loc[i, f'heating{suffix}'] = float(1)

        # Turn heating off if none of the conditions apply
        else:
            heating_active = False
            df.loc[i, f'heating{suffix}'] = float(0)

    return df

def compute_no_daydegrees(df, temp='temp', date_col='date', dd_col='dd'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Definice otopné sezóny – dny, kdy je měsíc v rozsahu září až květen
    df['in_heating_season'] = df[date_col].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5])
    df[dd_col] = 0.0                                        # Inicializace nového sloupce s denostupně na 0
    mask = (df['in_heating_season']) & (df[temp] <= 13)     # maska pro dny v otopní sezóně a teplota ≤ 13 °C
    df.loc[mask, dd_col] = 21 - df.loc[mask, temp]
    df.drop(columns=['in_heating_season'], inplace=True)
    
    return df

def shift_temperatures(df, temp='temp'):
    df[temp] = pd.to_numeric(df[temp])
    df = df.sort_values(by='date', ascending=True)

    df[temp + '_f_1d'] = df[temp].shift(-1) # Forward shift for the next day's temperature
    df[temp + '_f_2d'] = df[temp].shift(-2) # Forward shift for the next day's temperature
    df[temp + '_1d'] = df[temp].shift(1)
    df[temp + '_2d'] = df[temp].shift(2)
    df[temp + '_3d'] = df[temp].shift(3)
    df[temp + '_4d'] = df[temp].shift(4)
    df[temp + '_5d'] = df[temp].shift(5)
    df[temp + '_6d'] = df[temp].shift(6)
    df[temp + '_7d'] = df[temp].shift(7)
    return df

def shift_variable(df, var):
    df[var] = pd.to_numeric(df[var])
    df = df.sort_values(by='date', ascending=True)

    df[var + '_1d'] = df[var].shift(1)
    df[var + '_2d'] = df[var].shift(2)
    df[var + '_3d'] = df[var].shift(3)
    df[var + '_4d'] = df[var].shift(4)
    df[var + '_5d'] = df[var].shift(5)
    return df


if __name__ == '__main__':
    # prepare of datasets for training data
    # here one can have a look for construction
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format='%(asctime)s %(levelname)s | %(message)s')
    logging.info('')
    start_date = pd.to_datetime('2016-07-08')
    end_date   = pd.to_datetime('2024-11-25')

    df = get_train_data(start_date, end_date)
    print(list(df.columns))

    nan_only_columns = df.columns[df.isna().all()]
    print(nan_only_columns)

    nan_df = df.loc[df.isna().any(axis=1), df.columns[df.isna().any(axis=0)]]
    print(nan_df)

    nan_df.to_csv('_check_train_data.csv', index=True)


