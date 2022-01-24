from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

def interpolateTemperature(df):
  temperature = pd.DataFrame(df['temperature'])
  temperature.index = pd.to_datetime(df.index)
  temperature_interpolate = temperature.interpolate(method="linear")
  date = temperature.index
  interpolateAkima_temperature = pd.DataFrame(temperature_interpolate)
  return interpolateAkima_temperature

def interpolateHumidity(df):
  humidity = pd.DataFrame(df['humidity'])
  humidity.index = pd.to_datetime(df.index)
  humidity_interpolate = humidity.interpolate(method="linear")
  date = humidity.index
  interpolateLinear_humidity = pd.DataFrame(humidity_interpolate)
  return interpolateLinear_humidity

def interpolateUV(df):
  UV_index = pd.DataFrame(df['UV'])
  UV_index.index = pd.to_datetime(df.index)
  UV_index_interpolate = UV_index.interpolate(method="linear")
  date = UV_index.index
  interpolateLinear_UV_index = pd.DataFrame(UV_index_interpolate)
  return interpolateLinear_UV_index

def imputation_KNN(df):
  KNN_imputer=KNNImputer(n_neighbors=6)
  KNN_imputer.fit(df)
  X_knn= KNN_imputer.transform(df)
  performance = pd.DataFrame(X_knn)
  df1 = df.reset_index()
  result = pd.concat([df1, performance[[0,1,2]]], axis=1)
  result = result.drop(columns=['CO2', 'CO', 'PM2.5'])
  result = result.rename({0: 'CO2', 1: 'CO', 2: 'PM2.5'}, axis=1)
  result.index = result['index']
  result.index.name = None
  return result

def fill_missing(df):
  UV_outlier = interpolateUV(df)
  temperature_outlier = interpolateHumidity(df)
  humidity_outlier = interpolateTemperature(df)
  pollutants = df[['CO2','CO', 'PM2.5']]
  pollutants.index = pd.to_datetime(pollutants.index)
  pollutants_fill = pd.concat([pollutants, UV_outlier, temperature_outlier, humidity_outlier], axis = 1)
  fill_missing_result = imputation_KNN(pollutants_fill)
  print(fill_missing_result.isnull().sum())
  return fill_missing_result