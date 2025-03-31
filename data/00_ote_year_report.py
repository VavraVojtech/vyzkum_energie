import logging
import os
import sys
import pandas as pd
import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

# https://www.ote-cr.cz/cs/statistika/rocni-zprava?date=2019-01-01

def get_years(start_date, end_date):

    # get list of years to download
    years = list(range(start_date.year, end_date.year + 1))
    years = [str(year) for year in years]
    return years

def get_ote_url(year):
    # https://www.ote-cr.cz/pubweb/attachments/62_162/2019/Rocni_zprava_o_trhu_2019_V2_plyn.zip
    # https://www.ote-cr.cz/pubweb/attachments/62_162/2024/Rocni_zprava_o_trhu_2024_V2_plyn.zip
    url = f'https://www.ote-cr.cz/pubweb/attachments/62_162/{year}/' 
    url += f'Rocni_zprava_o_trhu_{year}_V2_plyn.zip'
    return url

def download_zipfile_and_create_df(year):
    # try:        
        url = get_ote_url(year)
        resp = urlopen(url)
        ZipFile(BytesIO(resp.read())).extractall('./extract')

        # get data from sheets
        if year < '2024': # migh there be exception for year 2024 and higher
            filename_xlsx = f'./extract/Rocni_zprava_o_trhu_plyn_{year}_V2.xls'
            df_trh_kladn = pd.read_excel(filename_xlsx, sheet_name='Trh s NFL +', header=1, skiprows=4) # Trh s NFL +
            df_trh_zapor = pd.read_excel(filename_xlsx, sheet_name='Trh s NFL -', header=1, skiprows=4) # Trh s NFL +
        else:
            filename_xlsx = f'./extract/Rocni_zprava_o_trhu_plyn_{year}_V2.xlsx'
            df_trh_kladn = pd.read_excel(filename_xlsx, sheet_name='Trh s NFL +', header=1, skiprows=4) # Trh s NFL +
            df_trh_zapor = pd.read_excel(filename_xlsx, sheet_name='Trh s NFL -', header=1, skiprows=4) # Trh s NFL +

        # os.remove(filename_xlsx)
        return df_trh_kladn, df_trh_zapor

    # except Exception as e:
    #     logging.error('ERROR: FAILED TO DOWNLOAD ZIPFILE AND CREATE DF. REASON: %s' % e)

def get_diagrams(start_date, end_date):

    years = get_years(start_date, end_date)
    df_out_trh_kladn = pd.DataFrame()
    df_out_trh_zapor = pd.DataFrame()
    for year in years:
        df_trh_kladn, df_trh_zapor = download_zipfile_and_create_df(year)
        # if int(year) >= 2019: # inconsistency in data
        #     df = df.drop(index=df.index[-4:])
        logging.info('DATA FOR YEAR ' + year + ' DOWNLOADED')
        df_out_trh_kladn = pd.concat([df_out_trh_kladn, df_trh_kladn])
        df_out_trh_zapor = pd.concat([df_out_trh_zapor, df_trh_zapor])
    return df_out_trh_kladn, df_out_trh_zapor

def arrange_df_trh_kladn(df, date_from, date_to):
    df.rename(columns={'Plynárenský den': 'DATE',
                       'Množství kladné nevyužité flexibility\n(MWh)': 'flex_mnozstvi_+',
                       'Zobchodovaná kladná nevyužitá flexibilita\n(MWh)': 'flex_obchod_+',
                       'Marginální cena kladné nevyužité flexibility\n(CZK/MWh)': 'flex_cena_+',
                       }, inplace=True)
    # delete last three rows with description
    df = df.dropna(axis=1, how='all')

     # make sure that all years are the same
    df.loc[:, 'DATE'] = pd.to_datetime(df['DATE'], format='%d.%m.%Y').dt.date
    df.set_index('DATE', inplace=True)
    df = df.loc[df.index >= date_from.normalize().date()].dropna()
    df = df.loc[df.index <= date_to.normalize().date()].dropna()
    return df

def arrange_df_trh_zapor(df, date_from, date_to):
    
    df.rename(columns={'Plynárenský den': 'DATE',
                       'Množství záporné nevyužité flexibility\n(MWh)': 'flex_mnozstvi_-',
                       'Zobchodovaná záporná nevyužitá flexibilita\n(MWh)': 'flex_obchod_-',
                       'Marginální cena záporné nevyužité flexibility\n(CZK/MWh)': 'flex_cena_-',
                       }, inplace=True)
    # delete last three rows with description
    df = df.dropna(axis=1, how='all')

     # make sure that all years are the same
    df.loc[:, 'DATE'] = pd.to_datetime(df['DATE'], format='%d.%m.%Y').dt.date
    df.set_index('DATE', inplace=True)
    df = df.loc[df.index >= date_from.normalize().date()].dropna()
    df = df.loc[df.index <= date_to.normalize().date()].dropna()
    return df

def write_to_output(df, table):
    """
    Write output to database
    Args:
        df: Output dataframe
        table: Table name
    """
    try:
        df.to_csv('output_data/' + table + '.csv', index=True)
        logging.info(table.upper() + ': SUCCESSFUL WRITE TO CSV.')
    except Exception as e:
        logging.error(table.upper() + ': FAILED TO SAVE TO CSV. REASON: %s' % e)

# def write_to_pickle(df):
#     PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
#     pickle_name = f'{PROJECT_ROOT}/pickled_data/temperatures.pickle'

#     try:
#         df.to_pickle(pickle_name)
#         logging.info('PICKLE write successful: ' + pickle_name)
#     except Exception as e:
#         logging.error(e)

def run_ote_report(date_from, date_to, table):
    df_trh_kladn, df_trh_zapor = get_diagrams(date_from, date_to)
    
    df_trh_kladn = arrange_df_trh_kladn(df_trh_kladn, date_from, date_to)
    df_trh_zapor = arrange_df_trh_zapor(df_trh_zapor, date_from, date_to)
    df = df_trh_kladn.merge(df_trh_zapor, on='DATE', how='left')

    logging.info(f'DATA FROM {date_from.date()} TO {date_to.date()} READY TO STORE')
    logging.info('\n' + str(df))
    write_to_output(df, table)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format='%(asctime)s %(levelname)s | %(message)s')

    # date_from = pd.to_datetime('today').normalize() - timedelta(days=1)  # or pd.to_datetime('2023-02-17')
    # date_to = pd.to_datetime('today')
    date_from = pd.to_datetime('2016-01-01') # below 2016 problems
    date_to = pd.to_datetime('2024-12-31')

    table = 'OTE_NG_REPORT'

    # execution
    # try:
    run_ote_report(date_from, date_to, table)
    # except Exception as e:
    #     logging.error('ERROR: FAILED TO EXECUTE THE SCRIPT. REASON: %s' % e)
