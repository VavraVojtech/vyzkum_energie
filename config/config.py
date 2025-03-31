import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# spotreba_cr

input_spotreba_cr = [
    'heating_OM', 'dd_OM', 'day_of_week', 'is_working_day', 'seasonality', 'posan', # 'month', 'season',  # 'is_weekend', 'is_holiday',
    'spotreba_cr_1d', # 'spotreba_cr_2d', 'spotreba_cr_3d', 'spotreba_cr_4d', 'spotreba_cr_5d',
    'roll_spotreba_cr_mean_2', # 'roll_spotreba_cr_mean_3', 'roll_spotreba_cr_mean_4',

    'temperature_2m', 'apparent_temperature', 'direct_radiation', 
    # 'relative_humidity_2m', # 'dew_point_2m', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 

    # 'mix_temperature_diff_00', 'mix_temperature_diff_11',
    # 'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

    'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'direct_radiation_diff_01',
    # 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01', 'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 
]

# 'flex_mnozstvi_+'
input_mnoz_plus = [
    'heating_OM', 'dd_OM', 'month', 'day_of_week', 'is_working_day', 'seasonality', 'season', 'posan', # 'is_weekend', 'is_holiday',
    'flex_mnozstvi_+_1d', 'flex_mnozstvi_+_2d', 'flex_mnozstvi_+_3d', 'flex_mnozstvi_+_4d', 'flex_mnozstvi_+_5d',
    'roll_flex_mnozstvi_+_mean_2', 'roll_flex_mnozstvi_+_mean_3', 'roll_flex_mnozstvi_+_mean_4',

'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'direct_radiation',

'mix_temperature_diff_00', 'mix_temperature_diff_11',
'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01',
'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 'direct_radiation_diff_01',
]

# 'flex_obchod_+'
input_obch_plus = [
    'heating_OM', 'dd_OM', 'month', 'day_of_week', 'is_working_day', 'seasonality', 'season', 'posan', # 'is_weekend', 'is_holiday',
    'flex_obchod_+_1d', 'flex_obchod_+_2d', 'flex_obchod_+_3d', 'flex_obchod_+_4d', 'flex_obchod_+_5d',
    'roll_flex_obchod_+_mean_2', 'roll_flex_obchod_+_mean_3', 'roll_flex_obchod_+_mean_4',

'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'direct_radiation',

'mix_temperature_diff_00', 'mix_temperature_diff_11',
'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01',
'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 'direct_radiation_diff_01',
]

# 'flex_cena_+'
input_cena_plus = [
    'heating_OM', 'dd_OM', 'month', 'day_of_week', 'is_working_day', 'seasonality', 'season', 'posan', # 'is_weekend', 'is_holiday',
    'flex_cena_+_1d', 'flex_cena_+_2d', 'flex_cena_+_3d', 'flex_cena_+_4d', 'flex_cena_+_5d',
    'roll_flex_cena_+_mean_2', 'roll_flex_cena_+_mean_3', 'roll_flex_cena_+_mean_4',

'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'direct_radiation',

'mix_temperature_diff_00', 'mix_temperature_diff_11',
'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01',
'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 'direct_radiation_diff_01',
]

# 'flex_mnozstvi_-'
input_mnoz_minu = [
    'heating_OM', 'dd_OM', 'month', 'day_of_week', 'is_working_day', 'seasonality', 'season', 'posan', # 'is_weekend', 'is_holiday',
    'flex_mnozstvi_-_1d', 'flex_mnozstvi_-_2d', 'flex_mnozstvi_-_3d', 'flex_mnozstvi_-_4d', 'flex_mnozstvi_-_5d',
    'roll_flex_mnozstvi_-_mean_2', 'roll_flex_mnozstvi_-_mean_3', 'roll_flex_mnozstvi_-_mean_4',

'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'direct_radiation',

'mix_temperature_diff_00', 'mix_temperature_diff_11',
'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01',
'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 'direct_radiation_diff_01',
]

# 'flex_obchod_-'
input_obch_minu = [
    'heating_OM', 'dd_OM', 'month', 'day_of_week', 'is_working_day', 'seasonality', 'season', 'posan', # 'is_weekend', 'is_holiday',
    'flex_obchod_-_1d', 'flex_obchod_-_2d', 'flex_obchod_-_3d', 'flex_obchod_-_4d', 'flex_obchod_-_5d',
    'roll_flex_obchod_-_mean_2', 'roll_flex_obchod_-_mean_3', 'roll_flex_obchod_-_mean_4',

'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'direct_radiation',

'mix_temperature_diff_00', 'mix_temperature_diff_11',
'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01',
'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 'direct_radiation_diff_01',
]

# 'flex_cena_-'
input_cena_minu = [
    'heating_OM', 'dd_OM', 'month', 'day_of_week', 'is_working_day', 'seasonality', 'season', 'posan', # 'is_weekend', 'is_holiday',
    'flex_cena_-_1d', 'flex_cena_-_2d', 'flex_cena_-_3d', 'flex_cena_-_4d', 'flex_cena_-_5d',
    'roll_flex_cena_-_mean_2', 'roll_flex_cena_-_mean_3', 'roll_flex_cena_-_mean_4',

'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'direct_radiation',

'mix_temperature_diff_00', 'mix_temperature_diff_11',
'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01',
'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 'direct_radiation_diff_01',
]





# list of fields:

list_of_all_fields = [
 # 'flex_mnozstvi_+'
'flex_mnozstvi_+_1d', 'flex_mnozstvi_+_2d', 'flex_mnozstvi_+_3d', 'flex_mnozstvi_+_4d', 'flex_mnozstvi_+_5d',
'roll_flex_mnozstvi_+_mean_2', 'roll_flex_mnozstvi_+_mean_3', 'roll_flex_mnozstvi_+_mean_4',

# 'flex_obchod_+'
'flex_obchod_+_1d', 'flex_obchod_+_2d', 'flex_obchod_+_3d', 'flex_obchod_+_4d', 'flex_obchod_+_5d',
'roll_flex_obchod_+_mean_2', 'roll_flex_obchod_+_mean_3', 'roll_flex_obchod_+_mean_4',

# 'flex_cena_+'
'flex_cena_+_1d', 'flex_cena_+_2d', 'flex_cena_+_3d', 'flex_cena_+_4d', 'flex_cena_+_5d',
'roll_flex_cena_+_mean_2', 'roll_flex_cena_+_mean_3', 'roll_flex_cena_+_mean_4',

# 'flex_mnozstvi_-'
'flex_mnozstvi_-_1d', 'flex_mnozstvi_-_2d', 'flex_mnozstvi_-_3d', 'flex_mnozstvi_-_4d', 'flex_mnozstvi_-_5d',
'roll_flex_mnozstvi_-_mean_2', 'roll_flex_mnozstvi_-_mean_3', 'roll_flex_mnozstvi_-_mean_4',

# 'flex_obchod_-'
'flex_obchod_-_1d', 'flex_obchod_-_2d', 'flex_obchod_-_3d', 'flex_obchod_-_4d', 'flex_obchod_-_5d',
'roll_flex_obchod_-_mean_2', 'roll_flex_obchod_-_mean_3', 'roll_flex_obchod_-_mean_4',

# 'flex_cena_-'
'flex_cena_-_1d', 'flex_cena_-_2d', 'flex_cena_-_3d', 'flex_cena_-_4d', 'flex_cena_-_5d',
'roll_flex_cena_-_mean_2', 'roll_flex_cena_-_mean_3', 'roll_flex_cena_-_mean_4',

# weather
'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m', 'direct_radiation',

'temperature_2m_f_1d', 'temperature_2m_f_2d',
'temperature_2m_1d', 'temperature_2m_2d', 'temperature_2m_3d', 'temperature_2m_4d', 'temperature_2m_5d', 'temperature_2m_6d', 'temperature_2m_7d',

'apparent_temperature_f_1d', 'apparent_temperature_f_2d',
'apparent_temperature_1d', 'apparent_temperature_2d', 'apparent_temperature_3d', 'apparent_temperature_4d', 'apparent_temperature_5d', 'apparent_temperature_6d', 'apparent_temperature_7d',

'direct_radiation_1d', 'direct_radiation_2d', 'direct_radiation_3d', 'direct_radiation_4d', 'direct_radiation_5d',
'cloud_cover_1d', 'cloud_cover_2d', 'cloud_cover_3d', 'cloud_cover_4d', 'cloud_cover_5d',
'wind_speed_10m_1d', 'wind_speed_10m_2d', 'wind_speed_10m_3d', 'wind_speed_10m_4d', 'wind_speed_10m_5d',
'relative_humidity_2m_1d', 'relative_humidity_2m_2d', 'relative_humidity_2m_3d', 'relative_humidity_2m_4d', 'relative_humidity_2m_5d',
'dew_point_2m_1d', 'dew_point_2m_2d', 'dew_point_2m_3d', 'dew_point_2m_4d', 'dew_point_2m_5d',
'wind_gusts_10m_1d', 'wind_gusts_10m_2d', 'wind_gusts_10m_3d', 'wind_gusts_10m_4d', 'wind_gusts_10m_5d',
'roll_temperature_2m_mean_2', 'roll_temperature_2m_mean_3', 'roll_temperature_2m_mean_4',

'mix_temperature_diff_00', 'mix_temperature_diff_11',
'mix_temperature_diff_22', 'mix_temperature_diff_00_11', 'mix_temperature_diff_00_22',

'temperature_2m_diff_01', 'apparent_temperature_diff_01', 'relative_humidity_2m_diff_01', 'dew_point_2m_diff_01',
'cloud_cover_diff_01', 'wind_speed_10m_diff_01', 'wind_gusts_10m_diff_01', 'direct_radiation_diff_01',

'temperature_2m_diff_02', 'apparent_temperature_diff_02', 'relative_humidity_2m_diff_02', 'dew_point_2m_diff_02',
'cloud_cover_diff_02', 'wind_speed_10m_diff_02', 'wind_gusts_10m_diff_02', 'direct_radiation_diff_02',

'heating_OM', 'dd_OM', 'month', 'day_of_week', 'is_weekend', 'is_holiday', 'is_working_day', 'seasonality', 'season', 'posan'
]