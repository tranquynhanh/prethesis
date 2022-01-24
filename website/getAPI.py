from flask import Blueprint, render_template
import requests
import numpy as np
import pandas as pd
import requests
import schedule
import time
from datetime import datetime, timedelta

Previous_Date = datetime.today() - timedelta(days=5)
start = Previous_Date.strftime ('%Y-%m-%d')
response = ("https://api.thingspeak.com/channels/1587927/feeds.json?api_key=0M7BCK9L3NHJZACV&timezone=Asia%2FBangkok&start={0}%2000:00:00".format(start))

def call_api():
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
    

# @getAPI.route('/')
# def getInfo():
    
#     return render_template('index.html', arr=arr)

# # schedule.every().day.at("00:00").do(call_api)

# while True:
#     schedule.run_pending()