import pandas as pd
import numpy as np
import logging
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from joblib import dump
import datetime

from config import config


def filter_data(df, start_date, end_date=None):
    if end_date is None:
        df = df[(df['date'] >= start_date)].copy()
    else:
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    return df

def save_model(model, model_name):
    filename = f'{config.PROJECT_ROOT}/model/{model_name}_{datetime.date.today()}.joblib'
    try:
        dump(model, filename)
        logging.info('Model write successful: ' + filename)
    except Exception as e:
        logging.error(e)

def write_compare_models_to_excel(df, suffix = 'what_ever_name'):
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f'{output_dir}/{suffix}.xlsx'
   
    try:
        df.to_excel(file_name, index=False)
        logging.info('File write successful: ' + file_name)
    except Exception as e:
        logging.error(e)
    return file_name