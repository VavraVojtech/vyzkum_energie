import logging
import pandas as pd
import numpy as np
import holidays
from datetime import timedelta

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import config

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_data():
    data = pd.read_csv('input_data/OTE_NG_REPORT.csv')
    data.rename(columns={'DATE': 'date'}, inplace=True)
    data_weather = pd.read_csv('input_data/OpenMeteo_train_data_6_6.csv')
    data = data.merge(data_weather, on='date', how='left')
    data.drop(columns=['visibility'], inplace=True)
    
    return data

def create_date_features(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['day_of_month'] = df[date_column].dt.day
    df['is_weekend'] = df[date_column].dt.weekday >= 5
    df['season'] = df[date_column].dt.month % 12 // 3 + 1
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df['season'] = df['season'].map(season_map)
    public_holidays = holidays.CountryHoliday('CZ')
    df['is_holiday'] = df[date_column].dt.date.isin(public_holidays.keys())

    return df

#### Now we cooking ###

def filter_data(df, start_date, end_date=None):
    if end_date is None:
        df = df[(df['date'] >= start_date)].copy()
    else:
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
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
            return 5 #'Autumn'
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

    return df[['date', f'heating{suffix}']]

def compute_no_daydegrees(df, temp='temp', date_col='date', dd_col='dd'):
    # https://www.chmi.cz/historicka-data/pocasi/otopna-sezona
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Definice otopné sezóny – dny, kdy je měsíc v rozsahu září až květen
    df['in_heating_season'] = df[date_col].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5])
    df[dd_col] = 0.0                                        # Inicializace nového sloupce s denostupně na 0
    mask = (df['in_heating_season']) & (df[temp] <= 13)     # maska pro dny v otopní sezóně a teplota ≤ 13 °C
    df.loc[mask, dd_col] = 21 - df.loc[mask, temp]
    df.drop(columns=['in_heating_season'], inplace=True)
    
    return df[['date', f'{dd_col}']]

def shift_variable(df, var):
    df[var] = pd.to_numeric(df[var])
    df = df.sort_values(by='date', ascending=True)

    df[var + '_1d'] = df[var].shift(1)
    df[var + '_2d'] = df[var].shift(2)
    df[var + '_3d'] = df[var].shift(3)
    df[var + '_4d'] = df[var].shift(4)
    df[var + '_5d'] = df[var].shift(5)
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

def get_rolling_stat(df, variable, num = 7):
    df[f'roll_{variable}_mean_{num}'] = df[variable].rolling(window=num).mean().shift(1)
    df[f'roll_{variable}_std_{num}'] = df[variable].rolling(window=num).std().shift(1)
    return df

def train_train_data(commodity, start_date, end_date):
    # get date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'date': date_range})
    df = set_date_features(df)

    # Add OM Weather Data
    if commodity == 'ng':
        df_weather_OM = pd.read_csv('input_data/OpenMeteo_train_data_6_6.csv')   # gas day
    elif commodity == 'pwr':
        df_weather_OM = pd.read_csv('input_data/OpenMeteo_train_data_0_24.csv')  # regular day
    df_weather_OM.drop(columns=['visibility'], inplace=True)
    
    # special treatment for temperature_2m
    df_weather_OM = shift_temperatures(df_weather_OM, 'temperature_2m')
    for num in range(2, 8):
        df_weather_OM = get_rolling_stat(df_weather_OM, 'temperature_2m', num=num)
    
    # special treatment for apparent_temperature
    df_weather_OM = shift_temperatures(df_weather_OM, 'apparent_temperature')
    for num in range(2, 8):
        df_weather_OM = get_rolling_stat(df_weather_OM, 'apparent_temperature', num=num)
    
    # OM shift other vars
    df_weather_OM = shift_variable(df_weather_OM, 'direct_radiation')
    df_weather_OM = shift_variable(df_weather_OM, 'cloud_cover')
    df_weather_OM = shift_variable(df_weather_OM, 'wind_speed_10m')
    df_weather_OM = shift_variable(df_weather_OM, 'relative_humidity_2m')
    df_weather_OM = shift_variable(df_weather_OM, 'dew_point_2m')
    df_weather_OM = shift_variable(df_weather_OM, 'wind_gusts_10m')
    df_weather_OM = shift_variable(df_weather_OM, 'precipitation')
    
    # merging OM Weather Data
    df = pd.merge(df, df_weather_OM, on='date', how='left')

    # add heating
    heating = determine_heating_period(df_weather_OM, start_date=None, end_date=None, temp = 'temperature_2m', suffix='OM')          # heating period
    heating = filter_data(heating, start_date, end_date)
    df = pd.merge(df, heating, on='date', how='left')

    # add daydegree
    daydegree_OTE = compute_no_daydegrees(df_weather_OM, temp='temperature_2m', date_col='date', dd_col='dd_OM')
    daydegree_OTE = shift_variable(daydegree_OTE, 'dd_OM')
    daydegree_OTE['dd_OM_diff_01'] = daydegree_OTE['dd_OM'] - daydegree_OTE['dd_OM_1d']
    daydegree_OTE = filter_data(daydegree_OTE, start_date, end_date)
    df = pd.merge(df, daydegree_OTE, on='date', how='left')



    return df



if __name__ == '__main__':
    target = 'flex_cena_+'
    features = ['temperature_2m','relative_humidity_2m','dew_point_2m',
                'apparent_temperature','cloud_cover','wind_speed_10m',
                'wind_gusts_10m','direct_radiation']
    
    data = get_data()
    
    # Vykreslení heatmapy
    corr_matrix = data.drop(columns=['date']).corr() # multicolinearity check
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Korelační matice")
    plt.savefig("graphs/correlation.png", dpi=300, bbox_inches='tight')
    print('Hotovo: graphs/correlation.png')

    # date adjust
    data = create_date_features(data, 'date')       # + add data
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    # timeseries graf
    fig, axes = plt.subplots(nrows=len(data.columns), ncols=1, figsize=(12, 3 * len(data.columns)))
    for ax, col in zip(axes, data.columns):
        ax.plot(data.index, data[col], label=col)
        ax.set_title(col)
        ax.set_xlabel("Date")
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("graphs/time_series_plots.png", dpi=300, bbox_inches='tight')
    print('Hotovo: graphs/time_series_plots.png')