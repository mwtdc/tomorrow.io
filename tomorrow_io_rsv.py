#!/usr/bin/python3.9
#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import os
import pathlib
import urllib
import urllib.parse
import warnings
from sys import platform
from time import sleep

import numpy as np
import optuna
import pandas as pd
import pymysql
import pyodbc
import requests
import xgboost as xgb
import yaml
from optuna.samplers import TPESampler
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

# У API ограничение запросов для бесплатной учетки 500/сутки и 25/час

start_time = datetime.datetime.now()
warnings.filterwarnings('ignore')

print('# Tomorrow_io Start! #', datetime.datetime.now())

# Общий раздел

# Настройки для логера
if platform == 'linux' or platform == 'linux2':
    logging.basicConfig(filename=('/var/log/log-execute/'
                                  'log_journal_tomorrow_io_rsv.log.txt'),
                        level=logging.INFO,
                        format=('%(asctime)s - %(levelname)s - '
                                '%(funcName)s: %(lineno)d - %(message)s'))
elif platform == 'win32':
    logging.basicConfig(filename=(f'{pathlib.Path(__file__).parent.absolute()}'
                                  f'/log_journal_tomorrow_io_rsv.log.txt'),
                        level=logging.INFO,
                        format=('%(asctime)s - %(levelname)s - '
                                '%(funcName)s: %(lineno)d - %(message)s'))

# Загружаем yaml файл с настройками
with open(
    f'{pathlib.Path(__file__).parent.absolute()}/settings.yaml', 'r'
          ) as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings['telegram'])
sql_settings = pd.DataFrame(settings['sql_db'])
pyodbc_settings = pd.DataFrame(settings['pyodbc_db'])
tomorrow_io_settings = pd.DataFrame(settings['tomorrow_io_api'])

# Функция отправки уведомлений в telegram на любое количество каналов
# (указать данные в yaml файле настроек)


def telegram(i, text):
    msg = urllib.parse.quote(str(text))
    bot_token = str(telegram_settings.bot_token[i])
    channel_id = str(telegram_settings.channel_id[i])

    retry_strategy = Retry(
        total=3,
        status_forcelist=[101, 429, 500, 502, 503, 504],
        method_whitelist=["GET", "POST"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    http.post(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_id}&text={msg}', timeout=10)

# Функция коннекта к базе Mysql
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)


def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    return pymysql.connect(host=host_yaml,
                           user=user_yaml,
                           port=port_yaml,
                           password=password_yaml,
                           database=database_yaml)

# Функция загрузки факта выработки
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)


def fact_load(i, dt):
    server = str(pyodbc_settings.host[i])
    database = str(pyodbc_settings.database[i])
    username = str(pyodbc_settings.user[i])
    password = str(pyodbc_settings.password[i])
    # Выбор драйвера в зависимости от ОС
    if platform == 'linux' or platform == 'linux2':
        connection_ms = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server}'
                                       ';SERVER=' + server + ';DATABASE='
                                       + database + ';UID='+username+';PWD='
                                       + password)
    elif platform == 'win32':
        connection_ms = pyodbc.connect('DRIVER={SQL Server};SERVER='
                                       + server + ';DATABASE=' + database +
                                       ';UID=' + username + ';PWD=' + password)
    #
    mssql_cursor = connection_ms.cursor()
    mssql_cursor.execute("SELECT SUBSTRING (Points.PointName ,"
                         "len(Points.PointName)-8, 8) as gtp, MIN(DT) as DT,"
                         " SUM(Val) as Val FROM Points JOIN PointParams ON "
                         "Points.ID_Point=PointParams.ID_Point JOIN PointMains"
                         " ON PointParams.ID_PP=PointMains.ID_PP WHERE "
                         "PointName like 'Генерация%{GVIE%' AND ID_Param=153 "
                         "AND DT >= " + str(dt) + " AND PointName NOT LIKE "
                         "'%GVIE0001%' AND PointName NOT LIKE '%GVIE0012%' "
                         "AND PointName NOT LIKE '%GVIE0416%' AND PointName "
                         "NOT LIKE '%GVIE0167%' AND PointName NOT LIKE "
                         "'%GVIE0264%' AND PointName NOT LIKE '%GVIE0007%' "
                         "AND PointName NOT LIKE '%GVIE0680%' AND PointName "
                         "NOT LIKE '%GVIE0987%' AND PointName NOT LIKE "
                         "'%GVIE0988%' AND PointName NOT LIKE '%GVIE0989%' "
                         "AND PointName NOT LIKE '%GVIE0991%' AND PointName "
                         "NOT LIKE '%GVIE0994%' AND PointName NOT LIKE "
                         "'%GVIE1372%' GROUP BY SUBSTRING (Points.PointName "
                         ",len(Points.PointName)-8, 8), DATEPART(YEAR, DT), "
                         "DATEPART(MONTH, DT), DATEPART(DAY, DT), "
                         "DATEPART(HOUR, DT) ORDER BY SUBSTRING "
                         "(Points.PointName ,len(Points.PointName)-8, 8), "
                         "DATEPART(YEAR, DT), DATEPART(MONTH, DT), "
                         "DATEPART(DAY, DT), DATEPART(HOUR, DT);")
    fact = mssql_cursor.fetchall()
    connection_ms.close()
    fact = pd.DataFrame(np.array(fact), columns=['gtp', 'dt', 'fact'])
    fact.drop_duplicates(subset=['gtp', 'dt'],
                         keep='last', inplace=True,
                         ignore_index=False)
    return fact

# Функция распаковки вложенного json


def json_extract(obj, key):
    'Рекурсивное извлечение значений из вложенного JSON.'
    arr = []

    def extract(obj, arr, key):
        'Рекурсивный поиск значений ключа в дереве JSON.'
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values

# Функция паузы
# нужна для ожидания следующего часа при превышении 25 запросов в часе


def pause():
    next_hour = datetime.datetime.now().hour + 1
    while True:
        if datetime.datetime.now().hour == next_hour:
            sleep(60)
            break

# Раздел загрузки прогноза погоды в базу


def load_forecast_to_db():
    telegram(1, "tomorrow_io: Старт загрузки погоды.")

    # Задаем переменные (даты для прогноза и список погодных параметров)
    api_key = tomorrow_io_settings.api_key[0]
    weather_dataframe = pd.DataFrame()
    col_parameters = ['temperature', 'temperatureApparent', 'dewPoint',
                      'humidity', 'windSpeed', 'windDirection',
                      'windGust', 'pressureSurfaceLevel',
                      'precipitationIntensity', 'rainIntensity',
                      'freezingRainIntensity', 'snowIntensity',
                      'sleetIntensity', 'precipitationProbability',
                      'precipitationType', 'rainAccumulation',
                      'snowAccumulation', 'snowAccumulationLwe',
                      'sleetAccumulation', 'sleetAccumulationLwe',
                      'iceAccumulation', 'iceAccumulationLwe',
                      'visibility', 'cloudCover', 'cloudBase',
                      'cloudCeiling', 'uvIndex', 'evapotranspiration',
                      'weatherCode']

    list_parameters = (','.join(col_parameters))
    #

    # Загрузка списка ГТП и координат из базы
    connection_geo = connection(1)
    with connection_geo.cursor() as cursor:
        sql = "select gtpp,lat,lng from treid_03.geo where gtpp like 'PV%';"
        cursor.execute(sql)
        ses_dataframe = pd.DataFrame(cursor.fetchall(),
                                     columns=['gtp', 'lat', 'lng'])
        connection_geo.close()
    logging.info('Список ГТП и координаты загружены из базы treid_03.geo')
    #

    # Загрузка прогнозов погоды по станциям
    final_dataframe = pd.DataFrame()
    g = 0
    f = 0
    for ses in range(len(ses_dataframe.index)):
        gtp = str(ses_dataframe.gtp[ses])
        lat = str(ses_dataframe.lat[ses]).replace(',', '.')
        lng = str(ses_dataframe.lng[ses]).replace(',', '.')
        print(gtp)
        try:
            # находим сколько часов осталось до конца суток
            # чтобы задать смещение по времени в запросе
            # нужно чтобы датафрейм всегда начинался с 00 часов следующих суток
            # иначе разное количество строк с часами при переходе с часа на час
            # при превышении 25 запросов в час
            starttm_offset_hrs = 24 - datetime.datetime.now().hour
            endtm_offset_hrs = starttm_offset_hrs + 23
            # print(starttime_offset_hours, endtime_offset_hours)
            #
            url_response = requests.get(f'https://api.tomorrow.io/v4/timelines'
                                        f'?location={lat},{lng}&'
                                        f'fields={list_parameters}&'
                                        f'units=metric&timesteps=1h&'
                                        f'timesteps=&startTime=nowPlus'
                                        f'{starttm_offset_hrs}h&'
                                        f'endTime=nowPlus{endtm_offset_hrs}h&'
                                        f'apikey={api_key}')
            url_response.raise_for_status()
            sleep(5)
            while url_response.status_code != 200:
                url_response = requests.get(f'https://api.tomorrow.io/v4/'
                                            f'timelines?location={lat},{lng}&'
                                            f'fields={list_parameters}&'
                                            f'units=metric&timesteps=1h&'
                                            f'timesteps=&startTime=nowPlus'
                                            f'{starttm_offset_hrs}h&'
                                            f'endTime=nowPlus{endtm_offset_hrs}'
                                            f'h&apikey={api_key}')
                sleep(20)
            if url_response.ok:
                weather_dataframe = pd.DataFrame()
                json_string = url_response.json()
                # print(json_string)
                weather_dataframe['startTime'] = json_extract(json_string,
                                                              'startTime')
                weather_dataframe.drop_duplicates(subset=['startTime'],
                                                  keep='last',
                                                  inplace=True,
                                                  ignore_index=True)
                # print(json_extract(json_string, 'uvIndex'))
                # print(weather_dataframe)
                for i in col_parameters:
                    # print(i)
                    weather_dataframe[i] = json_extract(json_string, i)
                    # print(weather_dataframe)
                weather_dataframe.insert(1, 'gtp', gtp)
                weather_dataframe.insert(
                    2, 'datetime_msc', (pd.to_datetime(
                        weather_dataframe['startTime'], utc=False
                        ) + pd.DateOffset(hours=3)))
                final_dataframe = final_dataframe.append(weather_dataframe,
                                                         ignore_index=True)
                # print(final_dataframe)
                g += 1
                f += 1
                print(g)
                print("прогноз погоды загружен")
                logging.info(
                    f'{g} Прогноз погоды для ГТП {gtp} загружен с tomorrow.io')
                if f == 25:
                    pause()
                    f = 0
            else:
                print(f'tomorrow_io: Ошибка запроса: '
                      f'{url_response.status_code}')
                logging.error(
                    f'tomorrow_io: Ошибка запроса: '
                    f'{url_response.status_code}')
                telegram(
                    1, f'tomorrow_io: Ошибка запроса: '
                       f'{url_response.status_code}')
                os._exit(1)
        except requests.HTTPError as http_err:
            print(f'tomorrow_io: HTTP error occurred: '
                  f'{http_err.response.text}')
            logging.error(
                f'tomorrow_io: HTTP error occurred: '
                f'{http_err.response.text}')
            telegram(
                1, f'tomorrow_io: HTTP error occurred: '
                   f'{http_err.response.text}')
            os._exit(1)
        except Exception as err:
            print(f'tomorrow_io: Other error occurred: {err}')
            logging.error(f'tomorrow_io: Other error occurred: {err}')
            telegram(1, f'tomorrow_io: Other error occurred: {err}')
            os._exit(1)
    final_dataframe.drop(['startTime'], axis='columns', inplace=True)
    final_dataframe['datetime_msc'] = final_dataframe[
        'datetime_msc'].astype('datetime64[ns]')
    final_dataframe.fillna(0, inplace=True)

    # преобразование из гтп потребления в гтп генерации
    final_dataframe['gtp'] = final_dataframe['gtp'].str.replace('P', 'G')

    gtp_dict = [('GVIE0011', ['GVIE0010']),
                ('GVIE0252', ['GVIE0602']),
                ('GVIE0676', ['GVIE0690']),
                ('GVIE0677', ['GVIE0678']),
                ('GVIE0694', ['GVIE0681']),
                ('GVIE0671', ['GVIE0682']),
                ('GVIE0687', ['GVIE0688']),
                ('GVIE0004', ['GVIE0002']),
                ('GVIE0233', ['GVIE0245']),
                ('GVIE0013', ['GVIE0247']),
                ('GVIE0340', ['GVIE0343']),
                ('GVIE1185', ['GVIE0427']),
                ('GVIE0455', ['GVIE0522']),
                ('GVIE0526', ['GVIE0528']),
                ('GVIE0123', ['GVIE0110', 'GVIE0111', 'GVIE0118', 'GVIE0120']),
                ('GVIE0112', ['GVIE0114', 'GVIE0115', 'GVIE0124']),
                ('GVIE0227', ['GVIE0229', 'GVIE1184']),
                ('GVIE0425', ['GVIE0417', 'GVIE0824', 'GVIE0825']),
                ('GVIE0836', ['GVIE0429', 'GVIE0603']),
                ('GVIE0023', ['GVIE0217', 'GVIE0008']),
                ('GVIE0695', ['GVIE0689', 'GVIE0691']),
                ]
    for pair in gtp_dict:
        temp = final_dataframe[final_dataframe.gtp == pair[0]]
        for x in pair[1]:
            temp.gtp = x
            final_dataframe = pd.concat([final_dataframe, temp], axis=0)
        final_dataframe = final_dataframe.sort_values(
            by=['gtp', 'datetime_msc']).reset_index(drop=True)

    final_dataframe.drop_duplicates(subset=['datetime_msc', 'gtp'],
                                    keep='last',
                                    inplace=True,
                                    ignore_index=False)
    final_dataframe.reset_index(drop=True, inplace=True)
    final_dataframe.to_excel(
        f'{pathlib.Path(__file__).parent.absolute()}/weather_dataframe.xlsx')
    # print(final_dataframe)
    telegram(1, f'tomorrow_io: загружен прогноз для {g} гтп')
    logging.info(f'Сформирован датафрейм для {g} гтп')
    #

    col_to_database = ['gtp', 'datetime_msc', 'loadtime', 'temperature',
                       'temperatureApparent', 'dewPoint',
                       'humidity', 'windSpeed', 'windDirection',
                       'windGust', 'pressureSurfaceLevel',
                       'precipitationIntensity', 'rainIntensity',
                       'freezingRainIntensity', 'snowIntensity',
                       'sleetIntensity', 'precipitationProbability',
                       'precipitationType', 'rainAccumulation',
                       'snowAccumulation', 'snowAccumulationLwe',
                       'sleetAccumulation', 'sleetAccumulationLwe',
                       'iceAccumulation', 'iceAccumulationLwe',
                       'visibility', 'cloudCover', 'cloudBase',
                       'cloudCeiling', 'uvIndex', 'evapotranspiration',
                       'weatherCode']

    list_col_database = (','.join(col_to_database))

    connection_om = connection(0)
    conn_cursor = connection_om.cursor()

    vall = ''
    rows = len(final_dataframe.index)
    gtp_rows = int(round(rows/24, 0))
    for r in range(len(final_dataframe.index)):
        vall = (vall+"('"
                + str(final_dataframe.gtp[r])+"','"
                + str(final_dataframe.datetime_msc[r])+"','"
                + str(datetime.datetime.now().isoformat())+"','"
                + str(final_dataframe.temperature[r])+"','"
                + str(final_dataframe.temperatureApparent[r])+"','"
                + str(final_dataframe.dewPoint[r])+"','"
                + str(final_dataframe.humidity[r])+"','"
                + str(final_dataframe.windSpeed[r])+"','"
                + str(final_dataframe.windDirection[r])+"','"
                + str(final_dataframe.windGust[r])+"','"
                + str(final_dataframe.pressureSurfaceLevel[r])+"','"
                + str(final_dataframe.precipitationIntensity[r])+"','"
                + str(final_dataframe.rainIntensity[r])+"','"
                + str(final_dataframe.freezingRainIntensity[r])+"','"
                + str(final_dataframe.snowIntensity[r])+"','"
                + str(final_dataframe.sleetIntensity[r])+"','"
                + str(final_dataframe.precipitationProbability[r])+"','"
                + str(final_dataframe.precipitationType[r])+"','"
                + str(final_dataframe.rainAccumulation[r])+"','"
                + str(final_dataframe.snowAccumulation[r])+"','"
                + str(final_dataframe.snowAccumulationLwe[r])+"','"
                + str(final_dataframe.sleetAccumulation[r])+"','"
                + str(final_dataframe.sleetAccumulationLwe[r])+"','"
                + str(final_dataframe.iceAccumulation[r])+"','"
                + str(final_dataframe.iceAccumulationLwe[r])+"','"
                + str(final_dataframe.visibility[r])+"','"
                + str(final_dataframe.cloudCover[r])+"','"
                + str(final_dataframe.cloudBase[r])+"','"
                + str(final_dataframe.cloudCeiling[r])+"','"
                + str(final_dataframe.uvIndex[r])+"','"
                + str(final_dataframe.evapotranspiration[r])+"','"
                + str(final_dataframe.weatherCode[r])+"'"+'),')

    vall = vall[:-1]
    sql = (f'INSERT INTO visualcrossing.tomorrow_io '
           f'({list_col_database}) VALUES {vall};')
    conn_cursor.execute(sql)
    connection_om.commit()
    connection_om.close()

    # Уведомление о записи в БД
    # telegram(0, f'tomorrow_io: записано в БД {rows} строк ({gtp_rows} гтп)')
    telegram(1, f'tomorrow_io: записано в БД {rows} строк ({gtp_rows} гтп)')
    logging.info(f'записано в БД {rows} строк c погодjq ({gtp_rows} гтп)')
    return final_dataframe

# Загрузка прогнозов погоды по станциям из базы и подготовка датафреймов


def prepare_datasets_to_train():
    col_in_database = ['gtp', 'datetime_msc', 'temperature',
                       'temperatureApparent', 'dewPoint',
                       'humidity', 'windSpeed', 'windDirection',
                       'windGust', 'pressureSurfaceLevel',
                       'precipitationIntensity', 'rainIntensity',
                       'freezingRainIntensity', 'snowIntensity',
                       'sleetIntensity', 'precipitationProbability',
                       'precipitationType', 'rainAccumulation',
                       'snowAccumulation', 'snowAccumulationLwe',
                       'sleetAccumulation', 'sleetAccumulationLwe',
                       'iceAccumulation', 'iceAccumulationLwe', 'visibility',
                       'cloudCover', 'cloudBase', 'cloudCeiling', 'uvIndex',
                       'evapotranspiration', 'weatherCode']

    list_col_database = (','.join(col_in_database))

    connection_geo = connection(0)
    with connection_geo.cursor() as cursor:
        sql = f'select gtp,def_power from visualcrossing.ses_gtp;'
        cursor.execute(sql)
        ses_dataframe = pd.DataFrame(cursor.fetchall(),
                                     columns=['gtp', 'def_power'])
        ses_dataframe['def_power'] = ses_dataframe['def_power']*1000
        ses_dataframe = ses_dataframe[ses_dataframe['gtp'].str.contains('GVIE', regex=False)]
        # ses_dataframe = ses_dataframe[(ses_dataframe['gtp'].str.contains('GVIE', regex=False)) | 
        #                               (ses_dataframe['gtp'].str.contains('GKZ', regex=False)) | 
        #                               (ses_dataframe['gtp'].str.contains('GROZ', regex=False))]
        connection_geo.close()
    # print(ses_dataframe)

    connection_forecast = connection(0)
    with connection_forecast.cursor() as cursor:
        sql = (f'select {list_col_database} from visualcrossing.tomorrow_io '
               f'where loadtime >= CURDATE() - INTERVAL 30 DAY;')
        cursor.execute(sql)
        forecast_dataframe = pd.DataFrame(cursor.fetchall(),
                                          columns=col_in_database)
        connection_forecast.close()
    logging.info('Загружен массив прогноза погоды за предыдущие дни')
    # Удаление дубликатов прогноза
    date_beg_predict = (
        datetime.datetime.today() + datetime.timedelta(days=1)
                        ).strftime("%Y-%m-%d")
    date_end_predict = (
        datetime.datetime.today() + datetime.timedelta(days=2)
                        ).strftime("%Y-%m-%d")
    forecast_dataframe.drop_duplicates(subset=['datetime_msc', 'gtp'],
                                       keep='last',
                                       inplace=True,
                                       ignore_index=False)
    forecast_dataframe['month'] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values).month
    forecast_dataframe['hour'] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values).hour

    test_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(
            forecast_dataframe['datetime_msc'] < str(date_beg_predict))[0]])
    test_dataframe.drop(
        forecast_dataframe.index[np.where(
            forecast_dataframe['datetime_msc'] > str(date_end_predict))[0]],
        inplace=True)
    test_dataframe = test_dataframe.merge(ses_dataframe,
                                          left_on=['gtp'],
                                          right_on=['gtp'],
                                          how='left')

    forecast_dataframe.drop(
        forecast_dataframe.index[np.where(
            forecast_dataframe['datetime_msc'] > str(datetime.datetime.today()))[0]],
        inplace=True)
    # Сортировка датафрейма по гтп и дате
    forecast_dataframe.sort_values(['gtp', 'datetime_msc'], inplace=True)
    forecast_dataframe['datetime_msc'] = forecast_dataframe[
        'datetime_msc'].astype('datetime64[ns]')
    logging.info('forecast и test dataframe преобразованы в нужный вид')

    #
    fact = fact_load(0, 'DATEADD(HOUR, -30 * 24, DATEDIFF(d, 0, GETDATE()))')
    # print(fact)
    #
    forecast_dataframe = forecast_dataframe.merge(ses_dataframe,
                                                  left_on=['gtp'],
                                                  right_on=['gtp'],
                                                  how='left')
    forecast_dataframe = forecast_dataframe.merge(fact,
                                                  left_on=['gtp',
                                                           'datetime_msc'],
                                                  right_on=['gtp', 'dt'],
                                                  how='left')
    # print(forecast_dataframe)
    # forecast_dataframe.to_excel('forecast_dataframe.xlsx')

    forecast_dataframe.dropna(subset=['fact'], inplace=True)
    forecast_dataframe.drop(['dt'], axis='columns', inplace=True)
    # print(forecast_dataframe)
    # print(test_dataframe)
    forecast_dataframe.to_excel('forecast_dataframe_rsv.xlsx')
    test_dataframe.to_excel('test_dataframe_rsv.xlsx')

    col_to_float = ['temperature', 'temperatureApparent', 'dewPoint',
                    'humidity', 'windSpeed', 'windDirection',
                    'windGust', 'pressureSurfaceLevel',
                    'precipitationIntensity', 'rainIntensity',
                    'freezingRainIntensity', 'snowIntensity',
                    'sleetIntensity', 'precipitationProbability',
                    'precipitationType', 'rainAccumulation',
                    'snowAccumulation', 'snowAccumulationLwe',
                    'sleetAccumulation', 'sleetAccumulationLwe',
                    'iceAccumulation', 'iceAccumulationLwe', 'visibility',
                    'cloudCover', 'cloudBase', 'cloudCeiling', 'uvIndex',
                    'evapotranspiration', 'def_power']
    for col in col_to_float:
        forecast_dataframe[col] = forecast_dataframe[col].astype('float')
        test_dataframe[col] = test_dataframe[col].astype('float')

    col_to_int = ['month', 'hour', 'weatherCode']
    for col in col_to_int:
        forecast_dataframe[col] = forecast_dataframe[col].astype('int')
        test_dataframe[col] = test_dataframe[col].astype('int')

    logging.info('Датафреймы погоды и факта выработки склеены')
    return forecast_dataframe, test_dataframe, col_to_float


# Раздел подготовки прогноза на XGBoost


def prepare_forecast_xgboost(forecast_dataframe, test_dataframe, col_to_float):
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe['fact'] == 0)])

    #

    # Загрузка прогноза за предыдущие дни
    # для расчета точности попадания и удаления строк с большой ошибкой
    # чтобы модель обучался только на том, где попала.

    # connection_predict = connection(1)
    # with connection_predict.cursor() as cursor:
    #     tomorrow_io_sql = ("SELECT gtp, dt, load_time, value 'tomorrow_io' "
    #                        "FROM treid_03.weather_foreca WHERE id_foreca IN "
    #                        "(22,23) AND DATE(load_time) = DATE_ADD(DATE(dt), "
    #                        "INTERVAL -1 DAY) AND DATE(dt) between "
    #                        "DATE_ADD(CURDATE(), INTERVAL -31 DAY) and "
    #                        "DATE_ADD(CURDATE(), INTERVAL 0 DAY) "
    #                        "ORDER BY gtp, dt;")
    #     cursor.execute(tomorrow_io_sql)
    #     tomorrow_io_dataframe = pd.DataFrame(
    #         cursor.fetchall(),
    #         columns=['gtp_tomorrow_io', 'dt_tomorrow_io',
    #                  'load_time_tomorrow_io', 'value_tomorrow_io'])
    #     tomorrow_io_dataframe.drop_duplicates(
    #         subset=['gtp_tomorrow_io', 'dt_tomorrow_io'],
    #         keep='last', inplace=True, ignore_index=False)
    #     # print(tomorrow_io_dataframe)
    #     tomorrow_io_dataframe.to_excel("tomorrow_io_dataframe.xlsx")
    # connection_predict.close()

    # z = z.merge(tomorrow_io_dataframe,
    #             left_on=['gtp',
    #                      'datetime_msc'],
    #             right_on=['gtp_tomorrow_io',
    #                       'dt_tomorrow_io'],
    #             how='left')

    # z.drop(['gtp_tomorrow_io', 'dt_tomorrow_io', 'load_time_tomorrow_io'],
    #        axis='columns', inplace=True)
    # z.dropna(subset=['value_tomorrow_io'], inplace=True)
    # z['score'] = (
    #     z['value_tomorrow_io'].astype('float') /
    #     z['fact'].astype('float'))
    # z.drop(z.index[np.where(z['score'] < 0.9)],
    #        inplace=True)
    # z.drop(z.index[np.where(z['score'] > 1.1)],
    #        inplace=True)
    # z.drop(['value_tomorrow_io', 'score'],
    #        axis='columns', inplace=True)
    #

    z['gtp'] = z['gtp'].str.replace('GVIE', '1')
    # z['gtp'] = z['gtp'].str.replace('GKZV', '4')
    # z['gtp'] = z['gtp'].str.replace('GKZ', '2')
    # z['gtp'] = z['gtp'].str.replace('GROZ', '3')
    x = z.drop(['fact', 'datetime_msc'], axis=1)
    # print(x)
    # x.to_excel("x.xlsx")
    y = z['fact'].astype('float')

    predict_dataframe = test_dataframe.drop(['datetime_msc'], axis=1)
    predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GVIE', '1')
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZV', '4')
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZ', '2')
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GROZ', '3')
    # print(predict_dataframe)
    # predict_dataframe.to_excel("predict_dataframe.xlsx")
    x['gtp'] = x['gtp'].astype('int')
    predict_dataframe['gtp'] = predict_dataframe['gtp'].astype('int')
    #
    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.8)
    logging.info('Старт предикта на XGBoostRegressor')

    param = {
        'lambda': 0.0010579265103973906,
        'alpha': 2.7386905767103067,
        'colsample_bytree': 0.3,
        'subsample': 0.5,
        'learning_rate': 0.012,
        'n_estimators': 10000,
        'max_depth': 9,
        'random_state': 2020,
        'min_child_weight': 1,
    }
    reg = xgb.XGBRegressor(**param)
    regr = BaggingRegressor(
        base_estimator=reg, n_estimators=3, n_jobs=-1).fit(x_train, y_train)
    # regr = reg.fit(x_train, y_train)
    predict = regr.predict(predict_dataframe)
    test_dataframe['forecast'] = pd.DataFrame(predict)

    # Важность фич, перед запуском и раскомменчиванием поменять regr выше.

    # feature_importance = reg.get_booster().get_score(importance_type='weight')
    # importance_df = pd.DataFrame()
    # importance_df['feature'] = pd.Series(feature_importance.keys())
    # importance_df['weight'] = pd.Series(feature_importance.values())
    # importance_df.sort_values(['weight'], inplace=True)
    # print(importance_df)

    logging.info('Подготовлен прогноз на XGBRegressor')
    #
    # Обработка прогнозных значений

    # Расчет исторического максимума выработки
    # для обрезки прогноза не по максимуму за месяц
    fact_historical = fact_load(0, '2015-04-01')
    fact_historical['month'] = pd.to_datetime(fact_historical.dt.values).month
    fact_historical['hour'] = pd.to_datetime(fact_historical.dt.values).hour
    gtp_dataframe = pd.DataFrame()
    for gtp in fact_historical.gtp.value_counts().index:
        gtp_month = fact_historical.loc[fact_historical.gtp == gtp]
        # print(gtp_month)
        for month in gtp_month.month.value_counts().index:
            max_month = gtp_month.loc[gtp_month.month == month,
                                      ['gtp', 'month', 'hour', 'fact']
                                      ].groupby(['hour'], as_index=False).max()
            # print(max_month)
            gtp_dataframe = gtp_dataframe.append(max_month, ignore_index=True)
            # print(gtp_dataframe)
    gtp_dataframe.sort_values(['gtp', 'month'], inplace=True)
    gtp_dataframe.reset_index(drop=True, inplace=True)
    gtp_dataframe = gtp_dataframe.reindex(
        columns=['gtp', 'month', 'hour', 'fact'])
    gtp_dataframe.fact[gtp_dataframe.fact < 50] = 0
    # print(gtp_dataframe)
    # gtp_dataframe.to_excel("max_month_dataframe.xlsx")
    #

    # Расчет максимума за месяц в часы
    max_month_dataframe = pd.DataFrame()
    date_cut = (
        datetime.datetime.today() + datetime.timedelta(days=-29)
        ).strftime('%Y-%m-%d')
    cut_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(
            forecast_dataframe['datetime_msc'] < str(date_cut))[0]])
    for gtp in test_dataframe.gtp.value_counts().index:
        max_month = cut_dataframe.loc[cut_dataframe.gtp == gtp,
                                      ['fact', 'hour', 'gtp']
                                      ].groupby(by=['hour']).max()
        max_month_dataframe = max_month_dataframe.append(max_month,
                                                         ignore_index=True)
    max_month_dataframe['hour'] = test_dataframe['hour']
    #

    test_dataframe = test_dataframe.merge(max_month_dataframe,
                                          left_on=['gtp', 'hour'],
                                          right_on=['gtp', 'hour'],
                                          how='left')
    test_dataframe = test_dataframe.merge(gtp_dataframe,
                                          left_on=['gtp', 'month', 'hour'],
                                          right_on=['gtp', 'month', 'hour'],
                                          how='left')
    # print(test_dataframe)
    # test_dataframe.to_excel('test_dataframe_2.xlsx')
    test_dataframe['fact'] = test_dataframe[
        ['fact_x', 'fact_y']].max(axis=1)
    test_dataframe.drop(['fact_x', 'fact_y'], axis='columns', inplace=True)
    test_dataframe['forecast'] = test_dataframe[
        ['forecast', 'fact', 'def_power']].min(axis=1)
    # Если прогноз отрицательный, то 0
    test_dataframe.forecast[test_dataframe.forecast < 0] = 0
    test_dataframe.drop(['fact', 'month', 'hour', 'weatherCode'],
                        axis='columns',
                        inplace=True)
    test_dataframe.drop(col_to_float, axis='columns', inplace=True)
    test_dataframe.to_excel(f'{pathlib.Path(__file__).parent.absolute()}/'
                            f'{(datetime.datetime.today() + datetime.timedelta(days=1)).strftime("%d.%m.%Y")}'
                            f'_xgboost_1day_orem_rsv.xlsx')
    logging.info(f'Датафрейм с прогнозом выработки прошел обработку'
                 f' от нулевых значений и обрезку по макс за месяц')
    #

    #
    # Запись прогноза в БД

    connection_vc = connection(1)
    conn_cursor = connection_vc.cursor()
    vall_predict = ''
    for p in range(len(test_dataframe.index)):
        vall_predict = (f"""{vall_predict}('{test_dataframe.gtp[p]}','"""
                        f"""{test_dataframe.datetime_msc[p]}','"""
                        f"""22','"""
                        f"""{datetime.datetime.now().isoformat()}','"""
                        f"""{round(test_dataframe.forecast[p], 3)}'),""")
    vall_predict = vall_predict[:-1]
    sql_predict = (f'INSERT INTO treid_03.weather_foreca '
                   f'(gtp,dt,id_foreca,load_time,value) '
                   f'VALUES {vall_predict};')
    # print(sql_predict)
    conn_cursor.execute(sql_predict)
    connection_vc.commit()
    connection_vc.close()

    #
    # Уведомление о подготовке прогноза
    # telegram(0, 'tomorrow_io: прогноз подготовлен')
    logging.info('Прогноз записан в БД treid_03')


def optuna_tune_params(forecast_dataframe, test_dataframe):
    # Подбор параметров через Optuna
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe['fact'] == 0)])

    #

    # Загрузка прогноза за предыдущие дни
    # для расчета точности попадания и удаления строк с большой ошибкой
    # чтобы модель обучался только на том, где попала.

    # connection_predict = connection(1)
    # with connection_predict.cursor() as cursor:
    #     tomorrow_io_sql = ("SELECT gtp, dt, load_time, value 'tomorrow_io' "
    #                        "FROM treid_03.weather_foreca WHERE id_foreca IN "
    #                        "(22,23) AND DATE(load_time) = DATE_ADD(DATE(dt), "
    #                        "INTERVAL -1 DAY) AND DATE(dt) between "
    #                        "DATE_ADD(CURDATE(), INTERVAL -31 DAY) and "
    #                        "DATE_ADD(CURDATE(), INTERVAL 0 DAY) "
    #                        "ORDER BY gtp, dt;")
    #     cursor.execute(tomorrow_io_sql)
    #     tomorrow_io_dataframe = pd.DataFrame(
    #         cursor.fetchall(),
    #         columns=['gtp_tomorrow_io', 'dt_tomorrow_io',
    #                  'load_time_tomorrow_io', 'value_tomorrow_io'])
    #     tomorrow_io_dataframe.drop_duplicates(
    #         subset=['gtp_tomorrow_io', 'dt_tomorrow_io'],
    #         keep='last', inplace=True, ignore_index=False)
    #     # print(tomorrow_io_dataframe)
    #     tomorrow_io_dataframe.to_excel("tomorrow_io_dataframe.xlsx")
    # connection_predict.close()

    # z = z.merge(tomorrow_io_dataframe,
    #             left_on=['gtp',
    #                      'datetime_msc'],
    #             right_on=['gtp_tomorrow_io',
    #                       'dt_tomorrow_io'],
    #             how='left')

    # z.drop(['gtp_tomorrow_io', 'dt_tomorrow_io', 'load_time_tomorrow_io'],
    #        axis='columns', inplace=True)
    # z.dropna(subset=['value_tomorrow_io'], inplace=True)
    # z['score'] = (
    #     z['value_tomorrow_io'].astype('float') /
    #     z['fact'].astype('float'))
    # z.drop(z.index[np.where(z['score'] < 0.9)],
    #        inplace=True)
    # z.drop(z.index[np.where(z['score'] > 1.1)],
    #        inplace=True)
    # z.drop(['value_tomorrow_io', 'score'],
    #        axis='columns', inplace=True)
    #

    z['gtp'] = z['gtp'].str.replace('GVIE', '1')
    # z['gtp'] = z['gtp'].str.replace('GKZV', '4')
    # z['gtp'] = z['gtp'].str.replace('GKZ', '2')
    # z['gtp'] = z['gtp'].str.replace('GROZ', '3')
    x = z.drop(['fact', 'datetime_msc'], axis=1)
    # print(x)
    # x.to_excel("x.xlsx")
    y = z['fact'].astype('float')

    predict_dataframe = test_dataframe.drop(['datetime_msc'], axis=1)
    predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GVIE', '1')
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZV', '4')
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GKZ', '2')
    # predict_dataframe['gtp'] = predict_dataframe['gtp'].str.replace('GROZ', '3')
    # print(predict_dataframe)
    # predict_dataframe.to_excel("predict_dataframe.xlsx")
    x['gtp'] = x['gtp'].astype('int')
    predict_dataframe['gtp'] = predict_dataframe['gtp'].astype('int')
    print(predict_dataframe.dtypes)

    def objective(trial):
        x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, train_size=0.8)
        # 'tree_method':'gpu_hist',
        # this parameter means using the GPU when training our model
        # to speedup the training process
        param = {
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical(
                'colsample_bytree',
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical(
                'subsample',
                [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_categorical(
                'learning_rate',
                [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
            'n_estimators': 10000,
            'max_depth': trial.suggest_categorical(
                'max_depth',
                [5, 7, 9, 11, 13, 15, 17]),
            'random_state': trial.suggest_categorical('random_state', [2020]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        }

        reg = xgb.XGBRegressor(**param)
        reg.fit(x_train, y_train, eval_set=[(x_validation, y_validation)],
                eval_metric='rmse', verbose=False, early_stopping_rounds=200)
        prediction = reg.predict(predict_dataframe)
        score = reg.score(x_train, y_train)
        return score

    study = optuna.create_study(sampler=TPESampler(), direction='maximize')
    study.optimize(objective, n_trials=1000, timeout=3600)
    optuna_vis = optuna.visualization.plot_param_importances(study)
    print(optuna_vis)
    print("Number of completed trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("\tBest Score: {}".format(trial.value))
    print("\tBest Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# Сам процесс работы разбит для удобства по функциям
# чтобы если погода загрузилась, а прогноз не подготовился,
#  то чтобы не тратить лимит запросов и не засорять базу,
# закомменчиваем первую функцию и разбираемся дальше сколько угодно попыток.
# 1 - load_forecast_to_db - загрузка прогноза с сайта и запись в бд
# 2 - prepare_datasets_to_train - подготовка датасетов для обучения модели,
# переменным присваиваются возвращаемые 2 датафрейма и список столбцов,
# необходимо для работы следующих функций.
# 3 - optuna_tune_params - подбор параметров для модели через оптуну
# необходимо в нее передать 2 датафрейма из предыдущей функции.
# 4 - prepare_forecast_xgboost - подготовка прогноза,
# в нее также необходимо передавать 2 датафрейма и список столбцов,
# который потом используется для удаления лишних столбцов,
# чтобы excel файл меньше места занимал.

# # 1
load_forecast_to_db()
# # 2
forecast_dataframe, test_dataframe, col_to_float = prepare_datasets_to_train()
# # 3
# optuna_tune_params(forecast_dataframe, test_dataframe)
# # 4
prepare_forecast_xgboost(forecast_dataframe, test_dataframe, col_to_float)

print('Время выполнения:', datetime.datetime.now() - start_time)
