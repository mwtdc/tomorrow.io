#!/usr/bin/python3.9
#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import pathlib
import urllib
import urllib.parse
import warnings
from sys import platform
from time import sleep

import numpy as np
import optuna
import pandas as pd
import pyodbc
import requests
import xgboost as xgb
import yaml
from optuna.samplers import TPESampler
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Коэффициент завышения прогноза:
OVERVALUE_COEFF = 1.0
# Задаем переменные (даты для прогноза и список погодных параметров)
DATE_BEG_PREDICT = (
    datetime.datetime.today() + datetime.timedelta(days=1)
).strftime("%Y-%m-%d")
DATE_END_PREDICT = (
    datetime.datetime.today() + datetime.timedelta(days=2)
).strftime("%Y-%m-%d")


COL_PARAMETERS = [
    "temperature",
    "temperatureApparent",
    "dewPoint",
    "humidity",
    "windSpeed",
    "windDirection",
    "windGust",
    "pressureSurfaceLevel",
    "precipitationIntensity",
    "rainIntensity",
    "freezingRainIntensity",
    "snowIntensity",
    "sleetIntensity",
    "precipitationProbability",
    "precipitationType",
    "rainAccumulation",
    "snowAccumulation",
    "snowAccumulationLwe",
    "sleetAccumulation",
    "sleetAccumulationLwe",
    "iceAccumulation",
    "iceAccumulationLwe",
    "visibility",
    "cloudCover",
    "cloudBase",
    "cloudCeiling",
    "uvIndex",
    "evapotranspiration",
    "weatherCode",
]
COL_TO_FLOAT = [
    "temperature",
    "temperatureApparent",
    "dewPoint",
    "humidity",
    "windSpeed",
    "windDirection",
    "windGust",
    "pressureSurfaceLevel",
    "precipitationIntensity",
    "rainIntensity",
    "freezingRainIntensity",
    "snowIntensity",
    "sleetIntensity",
    "precipitationProbability",
    "precipitationType",
    "rainAccumulation",
    "snowAccumulation",
    "snowAccumulationLwe",
    "sleetAccumulation",
    "sleetAccumulationLwe",
    "iceAccumulation",
    "iceAccumulationLwe",
    "visibility",
    "cloudCover",
    "cloudBase",
    "cloudCeiling",
    "uvIndex",
    "evapotranspiration",
    "def_power",
]
COL_IN_DATABASE = [
    "gtp",
    "datetime_msc",
    "loadtime",
] + COL_PARAMETERS

# У API ограничение запросов для бесплатной учетки 500/сутки и 25/час

start_time = datetime.datetime.now()
warnings.filterwarnings("ignore")

print("# Tomorrow_io Start! #", datetime.datetime.now())


# Настройки для логера
if platform == "linux" or platform == "linux2":
    logging.basicConfig(
        filename="/var/log/log-execute/log_journal_tomorrow_io_rsv.log.txt",
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(funcName)s: %(lineno)d - %(message)s"
        ),
    )
elif platform == "win32":
    logging.basicConfig(
        filename=(
            f"{pathlib.Path(__file__).parent.absolute()}"
            "/log_journal_tomorrow_io_rsv.log.txt"
        ),
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(funcName)s: %(lineno)d - %(message)s"
        ),
    )

# Загружаем yaml файл с настройками
with open(
    f"{pathlib.Path(__file__).parent.absolute()}/settings.yaml", "r"
) as yaml_file:
    settings = yaml.safe_load(yaml_file)
telegram_settings = pd.DataFrame(settings["telegram"])
sql_settings = pd.DataFrame(settings["sql_db"])
pyodbc_settings = pd.DataFrame(settings["pyodbc_db"])
tomorrow_io_settings = pd.DataFrame(settings["tomorrow_io_api"])

API_KEY = tomorrow_io_settings.api_key[0]


# Функция отправки уведомлений в telegram на любое количество каналов
# (указать данные в yaml файле настроек)
def telegram(i, text):
    try:
        msg = urllib.parse.quote(str(text))
        bot_token = str(telegram_settings.bot_token[i])
        channel_id = str(telegram_settings.channel_id[i])

        retry_strategy = Retry(
            total=3,
            status_forcelist=[101, 429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        http.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={channel_id}&text={msg}",
            verify=False,
            timeout=10,
        )
    except Exception as err:
        print(f"tomorrow_io: Ошибка при отправке в telegram -  {err}")
        logging.error(f"tomorrow_io: Ошибка при отправке в telegram -  {err}")


# Функция коннекта к базе Mysql
# (для выбора базы задать порядковый номер числом !!! начинается с 0 !!!!!)
def connection(i):
    host_yaml = str(sql_settings.host[i])
    user_yaml = str(sql_settings.user[i])
    port_yaml = int(sql_settings.port[i])
    password_yaml = str(sql_settings.password[i])
    database_yaml = str(sql_settings.database[i])
    db_data = f"mysql://{user_yaml}:{password_yaml}@{host_yaml}:{port_yaml}/{database_yaml}"
    return create_engine(db_data).connect()


# Функция загрузки факта выработки
def fact_load(i, dt):
    server = str(pyodbc_settings.host[i])
    database = str(pyodbc_settings.database[i])
    username = str(pyodbc_settings.user[i])
    password = str(pyodbc_settings.password[i])
    # Выбор драйвера в зависимости от ОС
    if platform == "linux" or platform == "linux2":
        connection_ms = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    elif platform == "win32":
        connection_ms = pyodbc.connect(
            "DRIVER={SQL Server};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
    #
    mssql_cursor = connection_ms.cursor()
    mssql_cursor.execute(
        "SELECT SUBSTRING (Points.PointName ,"
        "len(Points.PointName)-8, 8) as gtp, MIN(DT) as DT,"
        " SUM(Val) as Val FROM Points JOIN PointParams ON "
        "Points.ID_Point=PointParams.ID_Point JOIN PointMains"
        " ON PointParams.ID_PP=PointMains.ID_PP WHERE "
        "PointName like 'Генерация%{G%' AND ID_Param=153 "
        "AND DT >= "
        + str(dt)
        + " AND PointName NOT LIKE '%GVIE0001%' AND"
        ...
        " GROUP BY SUBSTRING (Points.PointName"
        " ,len(Points.PointName)-8, 8), DATEPART(YEAR, DT), DATEPART(MONTH,"
        " DT), DATEPART(DAY, DT), DATEPART(HOUR, DT) ORDER BY SUBSTRING"
        " (Points.PointName ,len(Points.PointName)-8, 8), DATEPART(YEAR, DT),"
        " DATEPART(MONTH, DT), DATEPART(DAY, DT), DATEPART(HOUR, DT);"
    )
    fact = mssql_cursor.fetchall()
    connection_ms.close()
    fact = pd.DataFrame(np.array(fact), columns=["gtp", "dt", "fact"])
    fact.drop_duplicates(
        subset=["gtp", "dt"], keep="last", inplace=True, ignore_index=False
    )
    fact["fact"] = fact["fact"].astype("float").round(-2)
    return fact


# Функция записи датафрейма в базу
def load_data_to_db(db_name, connect_id, dataframe):
    telegram(1, "tomorrow_io: Старт записи в БД.")
    logging.info("tomorrow_io: Старт записи в БД.")

    dataframe = pd.DataFrame(dataframe)
    connection_skm = connection(connect_id)
    try:
        dataframe.to_sql(
            name=db_name,
            con=connection_skm,
            if_exists="append",
            index=False,
            chunksize=5000,
        )
        rows = len(dataframe)
        telegram(
            1,
            f"tomorrow_io: записано в БД {rows} строк ({int(rows/24)} гтп)",
        )
        if len(dataframe.columns) > 30:
            telegram(
                0,
                f"tomorrow_io: записано в БД {rows} строк"
                f" ({int(rows/24)} гтп)",
            )
        logging.info(
            f"записано в БД {rows} строк c погодой ({int(rows/24)} гтп)"
        )
        telegram(1, "tomorrow_io: Финиш записи в БД.")
        logging.info("tomorrow_io: Финиш записи в БД.")
    except Exception as err:
        telegram(1, f"tomorrow_io: Ошибка записи в БД: {err}")
        logging.info(f"tomorrow_io: Ошибка записи в БД: {err}")


# Функция загрузки датафрейма из базы
def load_data_from_db(
    db_name,
    col_from_database,
    connect_id,
    condition_column,
    day_interval,
):
    telegram(1, "tomorrow_io: Старт загрузки из БД.")
    logging.info("tomorrow_io: Старт загрузки из БД.")

    list_col_database = ",".join(col_from_database)
    connection_db = connection(connect_id)
    if day_interval is None and condition_column is None:
        query = f"select {list_col_database} from {db_name};"
    else:
        query = (
            f"select {list_col_database} from {db_name} where"
            f" {condition_column} >= CURDATE() - INTERVAL {day_interval} DAY;"
        )
    dataframe_from_db = pd.read_sql(sql=query, con=connection_db)

    telegram(1, "tomorrow_io: Финиш загрузки из БД.")
    logging.info("tomorrow_io: Финиш загрузки из БД.")
    return dataframe_from_db


# Функция распаковки вложенного json
def json_extract(obj, key):
    "Рекурсивное извлечение значений из вложенного JSON."
    arr = []

    def extract(obj, arr, key):
        "Рекурсивный поиск значений ключа в дереве JSON."
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
def load_forecast_to_db(api_key, col_parameters, col_to_database):
    telegram(1, "tomorrow_io: Старт загрузки погоды.")

    # Задаем переменные (даты для прогноза и список погодных параметров)
    weather_dataframe = pd.DataFrame()
    list_parameters = ",".join(col_parameters)

    # Загрузка списка ГТП и координат из базы
    ses_dataframe = load_data_from_db(
        "treid_03.geo",
        ["gtpp", "lat", "lng"],
        1,
        None,
        None,
    )

    # Ниже можно выбирать гтп в датафрейме, только опт, кз, розн или все.
    ses_dataframe = ses_dataframe[
        ses_dataframe["gtpp"].str.contains("PV", regex=False)
    ]
    ses_dataframe.reset_index(inplace=True)
    logging.info("Список ГТП и координаты загружены из базы treid_03.geo")
    #

    # Загрузка прогнозов погоды по станциям
    final_dataframe = pd.DataFrame()
    g = 0
    f = 0
    for ses in range(len(ses_dataframe.index)):
        gtp = str(ses_dataframe.gtpp[ses])
        lat = str(ses_dataframe.lat[ses]).replace(",", ".")
        lng = str(ses_dataframe.lng[ses]).replace(",", ".")
        print(gtp)
        try:
            # находим сколько часов осталось до конца суток
            # чтобы задать смещение по времени в запросе
            # нужно чтобы датафрейм всегда начинался с 00 часов следующих суток
            # иначе разное количество строк с часами при переходе с часа на час
            # при превышении 25 запросов в час
            starttm_offset_hrs = 24 - datetime.datetime.now().hour
            endtm_offset_hrs = starttm_offset_hrs + 23

            url_response = requests.get(
                "https://api.tomorrow.io/v4/timelines"
                f"?location={lat},{lng}&"
                f"fields={list_parameters}&"
                "units=metric&timesteps=1h&"
                "timesteps=&startTime=nowPlus"
                f"{starttm_offset_hrs}h&"
                f"endTime=nowPlus{endtm_offset_hrs}h&"
                f"apikey={api_key}",
                verify=False,
            )

            sleep(5)
            while url_response.status_code != 200:
                url_response = requests.get(
                    "https://api.tomorrow.io/v4/"
                    f"timelines?location={lat},{lng}&"
                    f"fields={list_parameters}&"
                    "units=metric&timesteps=1h&"
                    "timesteps=&startTime=nowPlus"
                    f"{starttm_offset_hrs}h&"
                    f"endTime=nowPlus{endtm_offset_hrs}"
                    f"h&apikey={api_key}",
                    verify=False,
                )
                sleep(20)
            if url_response.ok:
                weather_dataframe = pd.DataFrame()
                json_string = url_response.json()
                weather_dataframe["startTime"] = json_extract(
                    json_string, "startTime"
                )
                weather_dataframe.drop_duplicates(
                    subset=["startTime"],
                    keep="last",
                    inplace=True,
                    ignore_index=True,
                )

                for i in col_parameters:
                    weather_dataframe[i] = json_extract(json_string, i)
                weather_dataframe.insert(1, "gtp", gtp)
                weather_dataframe.insert(
                    2,
                    "datetime_msc",
                    (
                        pd.to_datetime(
                            weather_dataframe["startTime"], utc=False
                        )
                        + pd.DateOffset(hours=3)
                    ),
                )
                weather_dataframe.insert(
                    3, "loadtime", datetime.datetime.now().isoformat()
                )
                final_dataframe = final_dataframe.append(
                    weather_dataframe, ignore_index=True
                )
                g += 1
                f += 1
                print(g)
                print("прогноз погоды загружен")
                logging.info(
                    f"{g} Прогноз погоды для ГТП {gtp} загружен с tomorrow.io"
                )
                if f == 25:
                    pause()
                    f = 0
            else:
                print(
                    f"tomorrow_io: Ошибка запроса: {url_response.status_code}"
                )
                logging.error(
                    f"tomorrow_io: Ошибка запроса: {url_response.status_code}"
                )
                telegram(
                    1,
                    f"tomorrow_io: Ошибка запроса: {url_response.status_code}",
                )
        except requests.HTTPError as http_err:
            print(
                f"tomorrow_io: HTTP error occurred: {http_err.response.text}"
            )
            logging.error(
                f"tomorrow_io: HTTP error occurred: {http_err.response.text}"
            )
            telegram(
                1,
                f"tomorrow_io: HTTP error occurred: {http_err.response.text}",
            )
        except Exception as err:
            print(f"tomorrow_io: Other error occurred: {err}")
            logging.error(f"tomorrow_io: Other error occurred: {err}")
            telegram(1, f"tomorrow_io: Other error occurred: {err}")

    final_dataframe.drop(["startTime"], axis="columns", inplace=True)
    final_dataframe["datetime_msc"] = final_dataframe["datetime_msc"].astype(
        "datetime64[ns]"
    )
    final_dataframe.fillna(0, inplace=True)

    # преобразование из гтп потребления в гтп генерации
    final_dataframe["gtp"] = final_dataframe["gtp"].str.replace("P", "G")

    gtp_dict = [
        ("GVIE0011", ["GVIE0010"]),
        ...
        ("GVIE0264", ["GVIE0680"]),
    ]
    for pair in gtp_dict:
        temp = final_dataframe[final_dataframe.gtp == pair[0]]
        for x in pair[1]:
            temp.gtp = x
            final_dataframe = pd.concat([final_dataframe, temp], axis=0)
        final_dataframe = final_dataframe.sort_values(
            by=["gtp", "datetime_msc"]
        ).reset_index(drop=True)

    final_dataframe.drop_duplicates(
        subset=["datetime_msc", "gtp"],
        keep="last",
        inplace=True,
        ignore_index=False,
    )
    final_dataframe.reset_index(drop=True, inplace=True)
    telegram(1, f"tomorrow_io: загружен прогноз для {g} гтп")
    logging.info(f"Сформирован датафрейм для {g} гтп")
    #

    load_data_to_db(
        "tomorrow_io",
        0,
        final_dataframe,
    )


# Загрузка прогнозов погоды по станциям из базы и подготовка датафреймов
def prepare_datasets_to_train(
    date_beg_predict, date_end_predict, col_in_database, col_to_float
):
    list_col_database = ",".join(col_in_database)

    ses_dataframe = load_data_from_db(
        "visualcrossing.ses_gtp",
        ["gtp", "def_power"],
        0,
        None,
        None,
    )
    ses_dataframe["def_power"] = ses_dataframe["def_power"] * 1000
    ses_dataframe = ses_dataframe[
        ses_dataframe["gtp"].str.contains("GVIE", regex=False)
    ]
    logging.info("Загружен датафрейм с гтп и установленной мощностью.")

    forecast_dataframe = load_data_from_db(
        "visualcrossing.tomorrow_io",
        col_in_database,
        0,
        "loadtime",
        365,
    )
    logging.info("Загружен массив прогноза погоды за предыдущие дни")
    # Удаление дубликатов прогноза
    forecast_dataframe.drop_duplicates(
        subset=["datetime_msc", "gtp"],
        keep="last",
        inplace=True,
        ignore_index=False,
    )
    forecast_dataframe["month"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).month
    forecast_dataframe["hour"] = pd.to_datetime(
        forecast_dataframe.datetime_msc.values
    ).hour

    test_dataframe = forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"] < str(date_beg_predict)
            )[0]
        ]
    )
    test_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"] > str(date_end_predict)
            )[0]
        ],
        inplace=True,
    )
    test_dataframe = test_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )

    forecast_dataframe.drop(
        forecast_dataframe.index[
            np.where(
                forecast_dataframe["datetime_msc"]
                > str(datetime.datetime.today())
            )[0]
        ],
        inplace=True,
    )
    # Сортировка датафрейма по гтп и дате
    forecast_dataframe.sort_values(["gtp", "datetime_msc"], inplace=True)
    forecast_dataframe["datetime_msc"] = forecast_dataframe[
        "datetime_msc"
    ].astype("datetime64[ns]")
    logging.info("forecast и test dataframe преобразованы в нужный вид")

    fact = fact_load(0, "DATEADD(HOUR, -365 * 24, DATEDIFF(d, 0, GETDATE()))")

    forecast_dataframe = forecast_dataframe.merge(
        ses_dataframe, left_on=["gtp"], right_on=["gtp"], how="left"
    )
    forecast_dataframe = forecast_dataframe.merge(
        fact,
        left_on=["gtp", "datetime_msc"],
        right_on=["gtp", "dt"],
        how="left",
    )

    forecast_dataframe.dropna(subset=["fact"], inplace=True)
    forecast_dataframe.drop(["dt"], axis="columns", inplace=True)

    for col in col_to_float:
        forecast_dataframe[col] = forecast_dataframe[col].astype("float")
        test_dataframe[col] = test_dataframe[col].astype("float")

    col_to_int = ["month", "hour", "weatherCode"]
    for col in col_to_int:
        forecast_dataframe[col] = forecast_dataframe[col].astype("int")
        test_dataframe[col] = test_dataframe[col].astype("int")

    logging.info("Датафреймы погоды и факта выработки склеены")
    return forecast_dataframe, test_dataframe


# Раздел подготовки прогноза на XGBoost
def prepare_forecast_xgboost(forecast_dataframe, test_dataframe, col_to_float):
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )
    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    x = z.drop(
        [
            "fact",
            "datetime_msc",
            "loadtime",
        ],
        axis=1,
    )
    y = z["fact"].astype("float")

    predict_dataframe = test_dataframe.drop(
        [
            "datetime_msc",
            "loadtime",
        ],
        axis=1,
    )
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    x["gtp"] = x["gtp"].astype("int")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")
    #
    x_train, x_validation, y_train, y_validation = train_test_split(
        x, y, train_size=0.9
    )
    logging.info("Старт предикта на XGBoostRegressor")

    param = {
        "lambda": 0.0010579265103973906,
        "alpha": 2.7386905767103067,
        "colsample_bytree": 0.3,
        "subsample": 0.5,
        "learning_rate": 0.012,
        "n_estimators": 10000,
        "max_depth": 9,
        "random_state": 2020,
        "min_child_weight": 1,
    }
    reg = xgb.XGBRegressor(**param)
    regr = BaggingRegressor(base_estimator=reg, n_estimators=3, n_jobs=-1).fit(
        x_train, y_train
    )
    # regr = reg.fit(x_train, y_train)
    predict = regr.predict(predict_dataframe)
    test_dataframe["forecast"] = pd.DataFrame(predict)
    test_dataframe["forecast"] = test_dataframe["forecast"] * OVERVALUE_COEFF

    logging.info("Подготовлен прогноз на XGBRegressor")
    #
    # Обработка прогнозных значений

    # Расчет исторического максимума выработки
    # для обрезки прогноза не по максимуму за месяц
    fact_historical = fact_load(0, "2015-04-01")
    fact_historical["month"] = pd.to_datetime(fact_historical.dt.values).month
    fact_historical["hour"] = pd.to_datetime(fact_historical.dt.values).hour
    gtp_dataframe = pd.DataFrame()
    for gtp in fact_historical.gtp.value_counts().index:
        gtp_month = pd.DataFrame(
            fact_historical.loc[fact_historical.gtp == gtp]
        )
        for month in gtp_month.month.value_counts().index:
            max_month = (
                gtp_month.loc[
                    gtp_month.month == month, ["gtp", "month", "hour", "fact"]
                ]
                .groupby(["hour"], as_index=False)
                .max()
            )
            gtp_dataframe = gtp_dataframe.append(max_month, ignore_index=True)

    gtp_dataframe.sort_values(["gtp", "month"], inplace=True)
    gtp_dataframe.reset_index(drop=True, inplace=True)
    gtp_dataframe = gtp_dataframe.reindex(
        columns=["gtp", "month", "hour", "fact"]
    )
    gtp_dataframe.fact[gtp_dataframe.fact < 50] = 0

    # Расчет максимума за месяц в часы
    max_month_dataframe = pd.DataFrame()
    date_cut = (
        datetime.datetime.today() + datetime.timedelta(days=-29)
    ).strftime("%Y-%m-%d")
    cut_dataframe = pd.DataFrame(
        forecast_dataframe.drop(
            forecast_dataframe.index[
                np.where(forecast_dataframe["datetime_msc"] < str(date_cut))[0]
            ]
        )
    )
    for gtp in test_dataframe.gtp.value_counts().index:
        max_month = (
            cut_dataframe.loc[
                cut_dataframe.gtp == gtp, ["fact", "hour", "gtp"]
            ]
            .groupby(by=["hour"])
            .max()
        )
        max_month.reset_index(inplace=True)
        max_month_dataframe = max_month_dataframe.append(
            max_month, ignore_index=True
        )

    test_dataframe = test_dataframe.merge(
        max_month_dataframe,
        left_on=["gtp", "hour"],
        right_on=["gtp", "hour"],
        how="left",
    )
    test_dataframe.fillna(0, inplace=True)
    test_dataframe = test_dataframe.merge(
        gtp_dataframe,
        left_on=["gtp", "month", "hour"],
        right_on=["gtp", "month", "hour"],
        how="left",
    )
    test_dataframe["fact"] = test_dataframe[["fact_x", "fact_y"]].min(axis=1)
    test_dataframe.drop(["fact_x", "fact_y"], axis="columns", inplace=True)
    test_dataframe["forecast"] = test_dataframe[
        ["forecast", "fact", "def_power"]
    ].min(axis=1)
    # Если прогноз отрицательный, то 0
    test_dataframe.forecast[test_dataframe.forecast < 0] = 0

    test_dataframe["forecast"] = np.where(
        test_dataframe["forecast"] == 0,
        (
            np.where(
                test_dataframe["fact"] > 0, np.NaN, test_dataframe.forecast
            )
        ),
        test_dataframe.forecast,
    )
    test_dataframe["forecast"].interpolate(
        method="linear", axis=0, inplace=True
    )
    test_dataframe["forecast"] = test_dataframe[["forecast", "fact"]].min(
        axis=1
    )

    test_dataframe.drop(
        [
            "fact",
            "month",
            "hour",
            "weatherCode",
            "loadtime",
        ],
        axis="columns",
        inplace=True,
    )
    test_dataframe.drop(col_to_float, axis="columns", inplace=True)
    # Добавить к датафрейму столбцы с текущей датой и id прогноза
    test_dataframe.insert(2, "id_foreca", "22")
    test_dataframe.insert(3, "load_time", datetime.datetime.now().isoformat())
    test_dataframe.rename(
        columns={"datetime_msc": "dt", "forecast": "value"},
        errors="raise",
        inplace=True,
    )
    logging.info(
        f"Датафрейм с прогнозом выработки прошел обработку"
        f" от нулевых значений и обрезку по макс за месяц"
    )
    #

    # Запись прогноза в БД
    load_data_to_db("weather_foreca", 1, test_dataframe)

    # Уведомление о подготовке прогноза
    telegram(0, "tomorrow_io: прогноз подготовлен")
    logging.info("Прогноз записан в БД treid_03")


def optuna_tune_params(forecast_dataframe, test_dataframe):
    # Подбор параметров через Optuna
    z = forecast_dataframe.drop(
        forecast_dataframe.index[np.where(forecast_dataframe["fact"] == 0)]
    )
    z["gtp"] = z["gtp"].str.replace("GVIE", "1")
    x = z.drop(["fact", "datetime_msc"], axis=1)
    y = z["fact"].astype("float")

    predict_dataframe = test_dataframe.drop(["datetime_msc"], axis=1)
    predict_dataframe["gtp"] = predict_dataframe["gtp"].str.replace(
        "GVIE", "1"
    )
    x["gtp"] = x["gtp"].astype("int")
    predict_dataframe["gtp"] = predict_dataframe["gtp"].astype("int")

    def objective(trial):
        x_train, x_validation, y_train, y_validation = train_test_split(
            x, y, train_size=0.8
        )
        param = {
            "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02],
            ),
            "n_estimators": 10000,
            "max_depth": trial.suggest_categorical(
                "max_depth", [5, 7, 9, 11, 13, 15, 17]
            ),
            "random_state": trial.suggest_categorical(
                "random_state", [500, 1000, 1500, 2000]
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        }

        reg = xgb.XGBRegressor(**param)
        reg.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=200,
        )
        prediction = reg.predict(predict_dataframe)
        score = reg.score(x_train, y_train)
        return score

    study = optuna.create_study(sampler=TPESampler(), direction="maximize")
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
# в нее также необходимо передавать 2 датафрейма

# # 1
load_forecast_to_db(API_KEY, COL_PARAMETERS, COL_IN_DATABASE)
# # 2
forecast_dataframe, test_dataframe = prepare_datasets_to_train(
    DATE_BEG_PREDICT, DATE_END_PREDICT, COL_IN_DATABASE, COL_TO_FLOAT
)
# # 3
# optuna_tune_params(forecast_dataframe, test_dataframe)
# # 4
prepare_forecast_xgboost(forecast_dataframe, test_dataframe, COL_TO_FLOAT)

print("Время выполнения:", datetime.datetime.now() - start_time)

