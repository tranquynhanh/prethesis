from threading import Timer
# from website import create_app
from flask import Blueprint, render_template, Flask, request
import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import os
from sklearn.preprocessing import MinMaxScaler
from fillMissing import *
from detect_outlier import *
from database import *
from predict import predict
from datetime import datetime

app = Flask(__name__)

Previous_Date = datetime.today() - timedelta(days=2)
start = Previous_Date.strftime ('%Y-%m-%d')
response = ("https://api.thingspeak.com/channels/1587927/feeds.json?api_key=0M7BCK9L3NHJZACV&timezone=Asia%2FBangkok&results=20")
response_day = "https://api.thingspeak.com/channels/1587927/feeds.json?api_key=0M7BCK9L3NHJZACV&timezone=Asia%2FBangkok&start={0}%2000:00:00".format(start)

CO2 = 0
CO = 0
PM25 = 0
UV = 0
temperature = 0
humidity = 0
MYBUG = True

def preprocessing_day():
    df = call_api(response_day)
    date_format_str = '%Y-%m-%d %H:%M:%S'
    length = len(df.date)-1
    for i in range(length):
        start = datetime.strptime(df.date[i], date_format_str)
        end =   datetime.strptime(df.date[i+1], date_format_str)
        diff = end - start  
        if (diff.total_seconds() > 330):
          add = start + timedelta(seconds=319)
          add = add.strftime(date_format_str)
          line = pd.DataFrame({"date": add ,"CO2": float("NaN"), "CO": float("NaN"),"PM2.5": float("NaN"), "UV": float("NaN"), "temperature": float("NaN"), "humidity": float("NaN")}, index=[i+1])
          df = pd.concat([df.iloc[:i+1], line, df.iloc[i+1:]]).reset_index(drop=True)
          length += 1
          continue
    df.index = df['date']
    df.index.name = None
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df_fill = fill_missing(df)
    df_detect = detect_anamoly(df_fill)
    df_detect_fill = fill_missing(df_detect)
    df_detect_fill = df_detect_fill.reindex(['index','CO2', 'CO','PM2.5','UV','temperature','humidity'], axis=1)
    return df_detect_fill

def predict_func(data, column, n_lag, n_predict):
    sc = MinMaxScaler()
    data = pd.DataFrame(data)
    data.index = data['date']
    data.index.name = None
    data_pred = data[column]
    data_pred = data_pred.reset_index()
    year = []
    month = []
    day = []
    hour = [] 
    minutes = []
    for i in range(len(data_pred['index'])):
      time = datetime.strptime(str(data_pred['index'][i]), "%Y-%m-%d %H:%M:%S")
      year.append(time.year)
      month.append(time.month)
      day.append(time.day)
      hour.append(time.hour)
      minutes.append(time.minute)
    data_pred['year'] = pd.DataFrame(year)[0]
    data_pred['month'] = pd.DataFrame(month)[0]
    data_pred['day'] = pd.DataFrame(day)[0]
    data_pred['hour'] = pd.DataFrame(hour)[0]
    data_pred['minute'] = pd.DataFrame(minutes)[0]
    for i in range(len(data_pred['minute'])):
        mode = data_pred['minute'][i]%5
        if (mode == 0):
          data_pred['minute'][i] = data_pred['minute'][i]
        else:
          data_pred['minute'][i] = data_pred['minute'][i] - mode
    data_pred = data_pred.groupby(['year', 'month','day','hour']).agg(
    {
        column :'mean',
    }
)
    data_pred = data_pred.reset_index()
    data_pred['minute'] = str("00")
    data_pred['second'] = str("00")
    cols = ['hour', 'minute', 'second']
    data_pred['time'] = data_pred[cols].apply(lambda x: ':'.join(x.values.astype(str)), axis="columns")
    for i in range(len(data_pred['time'])):
        if (len(data_pred['time'][i]) == 7):
            data_pred['time'][i] = "0" + data_pred['time'][i][0:]
    cols = ['year', 'month', 'day']
    data_pred['date'] = data_pred[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    data_pred['date'] = pd.to_datetime(data_pred['date'] + " "+ data_pred['time'])
    data_pred.index = data_pred['date']
    data_pred.index.name = None
    data_pred_element = data_pred[column]
    data_pred_element.index = data_pred.index
    data_pred_element = pd.DataFrame(data_pred_element)
    last_date = data_pred_element.index.values[-1]
    last_date = pd.date_range(last_date, periods=24, freq='H')
    predict_time = pd.DataFrame(index=[pd.Series(last_date)])
    scale = pd.DataFrame(sc.fit_transform(data_pred_element), index=data_pred_element.index, columns=data_pred_element.columns)
    deep_learner = predict(
        data = scale,
        Y_var = column,
        lag = n_lag,
        LSTM_layer_depth=48,
        epochs=200,
        train_test_split=0.1
    )

    deep_learner.LSTModel()
    yhat = deep_learner.predict_n_ahead(n_predict)
    yhat = [y[0][0] for y in yhat]
    yhat = np.asarray(yhat)
    yhat = sc.inverse_transform(yhat.reshape(-1,1))
    yhat = pd.DataFrame(yhat)
    yhat.index = predict_time.index
    print(yhat)
    yhat = yhat.rename({0: column}, axis=1)
    return yhat

def addToDB():
    data = preprocessing_day()
    data = data.rename({'index': 'date', 'CO2': 'co2', 'CO': 'co', 'PM2.5': 'pm', 'UV': 'uv', 'temperature': 'temperature', 'humidity': 'humidity'}, axis=1) 
    insert(data, "airactual")
    column_names = ['date', 'CO2', 'CO', 'PM2.5', 'UV', 'temperature', 'humidity']
    df = select("SELECT *  from airactual where date > current_timestamp - '5 days'::interval;", column_names)
    predict_CO2 = predict_func(df, "CO2", 24, 24)
    predict_CO = predict_func(df, "CO", 24,24)
    predict_PM = predict_func(df, "PM2.5", 24, 24)
    predict_UV = predict_func(df, "UV", 48, 24)
    predict_temperature = predict_func(df, "temperature", 48, 24)
    predict_humidity = predict_func(df, "humidity", 24, 24)
    predict_db = pd.concat([predict_CO2, predict_CO, predict_PM, predict_UV, predict_temperature, predict_humidity], axis = 1)
    predict_db = predict_db.reset_index()
    predict_db = predict_db.rename({'level_0': 'date', 'CO2': 'co2_pred', 'CO': 'co_pred', 'PM2.5': 'pm_pred', 'UV': 'uv_pred', 'temperature': 'temperature_pred', 'humidity': 'humidity_pred'}, axis=1) 
    insert(predict_db, "airpred")

def seeHistory(query):
    column_names = ['date', 'CO2', 'CO', 'PM2.5', 'UV', 'temperature', 'humidity']
    df = select(query, column_names)
    return df

def seeFuture(query, element):
    column_names = ['date', 'CO2', 'CO', 'PM2.5', 'UV', 'temperature', 'humidity']
    df = select(query, column_names)
    df = df.sort_values(by="date")
    df = pd.DataFrame(df)
    date = np.array(df['date'].astype(str).str[:10])
    time = np.array(df['date'].astype(str).str[11:16])  
    data_element = np.array(df[element])
    return date, data_element, time

# seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "CO2")

if not MYBUG or os.environ.get('WERKZEUG_RUN_MAIN') == 'true': 
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(addToDB,'cron', hour=18, minute=23)
    sched.start()

def call_api(response):
    get_data = requests.get(response).json()
    id = get_data['feeds']
    CO2 = []
    CO = []
    pm25 = []
    UV = []
    temperature = []
    humidity = []
    date = []
    for x in id: 
        date.append(x['created_at'])
        CO2.append(x['field1'])
        CO.append(x['field2'])
        pm25.append(x['field3'])
        UV.append(x['field4'])
        temperature.append(x['field5'])
        humidity.append(x['field6'])
    arr = np.stack((date, CO2, CO, pm25, UV, temperature, humidity), axis=1)
    df = pd.DataFrame(arr)
    df = df.rename({0: 'date', 1: 'CO2', 2: 'CO', 3: 'PM2.5', 4: 'UV', 5: 'temperature', 6: 'humidity'}, axis=1)
    df['date'] = df['date'].str[:-6]
    for i in range(len(df['date'])):
        df['date'][i] = df['date'][i].replace("T", " ")   
    return df

def get_data(response, type):
    df = call_api(response)
    CO2 = round(float(df.iloc[-1]['CO2']), 2)
    CO = round(float(df.iloc[-1]['CO']), 2)
    PM25 = round(float(df.iloc[-1]['PM2.5'])*1000, 2)
    UV = round(float(df.iloc[-1]['UV']), 2)
    temperature = int(float(df.iloc[-1]['temperature']))
    humidity = int(float(df.iloc[-1]['humidity']))
    df['date'] = df['date'].str[-9:]
    if type == "CO2":
        labels = df.iloc[-10:]['date'].to_numpy().tolist()
        values = df.iloc[-10:]['CO2'].astype(float).to_numpy().tolist()
    elif type == "CO":
        labels = df.iloc[-10:]['date'].to_numpy().tolist()
        values = df.iloc[-10:]['CO'].astype(float).to_numpy().tolist()
    elif type == "pm25":
        labels = df.iloc[-10:]['date'].to_numpy().tolist()
        s = df.iloc[-10:]['PM2.5']
        change = pd.to_numeric(s, errors='coerce')*1000
        values = change.astype(float).to_numpy().tolist()
        print(values)
    elif type == "UV":
        labels = df.iloc[-10:]['date'].to_numpy().tolist()
        values = df.iloc[-10:]['UV'].astype(float).to_numpy().tolist()
    elif type == "temperature-select":
        labels = df.iloc[-10:]['date'].to_numpy().tolist()
        values = df.iloc[-10:]['temperature'].astype(float).to_numpy().tolist()
    elif type == "humidity":
        labels = df.iloc[-10:]['date'].to_numpy().tolist()
        values = df.iloc[-10:]['humidity'].astype(float).to_numpy().tolist()
    return CO2, CO, PM25, UV, temperature,humidity, labels, values

@app.route('/', methods=['GET','POST'])  
def getInfo():
    date = ""
    pollutant = "CO2"
    date_pred, data_pred, time_pred = seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "CO2")
    data_pred = np.round(data_pred,1)
    CO2, CO, PM25, UV, temperature, humidity,labels, values = get_data(response, "CO2")
    if (
        request.method == "POST"
        and request.form['action'] == 'CO2'
    ):
        pollutant = "CO2"
        CO2, CO, PM25, UV, temperature, humidity,labels, values = get_data(response, "CO2")
        date_pred, data_pred, time_pred = seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "CO2")
        data_pred = np.round(data_pred,1)
    elif (
        request.method == "POST"
        and request.form['action'] == 'co'
    ):
        pollutant = "CO"
        CO2, CO, PM25, UV, temperature, humidity,labels, values = get_data(response, "CO")
        date_pred, data_pred, time_pred = seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "CO")
        data_pred = np.round(data_pred,1)
    elif (
        request.method == "POST"
        and request.form['action'] == 'pm25'
    ):
        pollutant = "PM2.5"
        CO2, CO, PM25, UV, temperature, humidity,labels, values = get_data(response, "pm25")
        date_pred, data_pred, time_pred = seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "PM2.5")
        data_pred = data_pred*1000
        data_pred = np.round(data_pred,1)
    elif (
        request.method == "POST"
        and request.form['action'] == 'UV'
    ):
        pollutant = "UV"
        CO2, CO, PM25, UV, temperature, humidity,labels, values = get_data(response, "UV")
        date_pred, data_pred, time_pred = seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "UV") 
        data_pred = np.round(data_pred,1)
    elif (
        request.method == "POST"
        and request.form['action'] == 'temperature-select'
    ):
        pollutant = "temperature"
        CO2, CO, PM25, UV, temperature, humidity, labels, values = get_data(response, "temperature-select")
        date_pred, data_pred, time_pred = seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "temperature")
        data_pred = np.round(data_pred,1)
    elif (
        request.method == "POST"
        and request.form['action'] == 'humidity'
    ):
        pollutant = "humidity"
        CO2, CO, PM25, UV, temperature, humidity,labels, values = get_data(response, "humidity")
        date_pred, data_pred, time_pred = seeFuture("SELECT * FROM airpred ORDER BY date DESC LIMIT 24", "humidity")  
        data_pred = np.round(data_pred,1)
    elif (
        request.method == "POST"
        and request.form["action"] == 'submit'
    ):
        date = request.form['date-selector']
        year = date[0:4]
        month = date[5:7]
        day = date[8:]
        query = "SELECT date, co2, co, pm, uv, temperature, humidity from airactual where extract(day from date) = {0} and extract(year from date) = {1} and extract(month from date) = {2};".format(day, year, month)
        df = seeHistory(query)
        date_format_str = '%Y-%m-%d %H:%M:%S'
        length = len(df.date)-1
        for i in range(length):
            df['date'][i] = df['date'][i].strftime(date_format_str)
        df['date'] = df['date'].astype(str).str[-9:] 
        labels = df.iloc[:]['date'].to_numpy().tolist()
        values = df.iloc[:]['CO2'].astype(float).to_numpy().tolist()
        pollutant = "CO2"

    return render_template(
        'index.html', 
        CO2=CO2, 
        CO=CO, 
        PM25=PM25, 
        UV = UV, 
        temperature=temperature,   
        humidity=humidity, 
        labels=labels, 
        values=values, 
        pollutant=pollutant, 
        date_pred=date_pred, 
        data_pred=data_pred,
        time_pred=time_pred
    )    

if __name__ == '__main__':
    app.run(debug=MYBUG)
