import numpy as np
import warnings
import itertools
import pandas
import math
import sys
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

class ARMA:
    def __init__(self, p,q):
        self.p = p
        self.q = q
    
    # Setters
    def set_p(self,p):
        self.p=p 
    
    def set_q(self,q):
        self.q=q
    
    def set_training_data_time(self, time):
        self.training_data_time = time
    
    def set_validation_data_time(self, time):
        self.validation_data_time = time
        
    def set_testing_data_time(self, time):
        self.testing_data_time = time
    
    def set_validation_data_set(self,data):
        self.validation_data_set = data
        
    def set_testing_data_set(self,data):
        self.testing_data_set = data
    
    def set_training_data_set(self,data):
        self.training_data = data
        self.Z = data - np.mean(data)
        self.Z.shape = (len(data),1)
    
    # Model
    def shock(self,mean,std):
        return np.random.normal(mean, std, 1)
    
    def calculate_AR_normal_matrix_x_row(self,data,t,mean,std):
        row = np.zeros((1,self.p+1))
        j = 0
        for i in range(t-self.p,t):
            if i < 0:
                row[0][j] = 0
            else:
                row[0][j] = data[i]
            j+=1
        row[0][-1] = self.shock(mean,std)
        return row
    
    def calculate_AR_weights(self):
        normal_matrix = np.zeros((len(self.training_data),self.p+1))
        mean = np.mean(self.Z)
        std = np.std(self.Z, ddof=1)
        for i in range(0,len(self.training_data)):
            normal_matrix[i] = self.calculate_AR_normal_matrix_x_row(self.Z,i,mean,std)
        
        normal_matrix_tanspose = normal_matrix.transpose()
        self.AR_weights = np.dot(np.dot(np.linalg.pinv(np.dot(normal_matrix_tanspose,normal_matrix)),normal_matrix_tanspose),self.Z)
        
    def get_AR_prediction(self,data_set):
        self.calculate_AR_weights()
        self.AR_prediction = np.zeros((np.max(data_set.shape),1))
        mean = np.mean(data_set)
        std = np.std(data_set, ddof=1)
        Z = np.array(data_set)
        Z.shape = (np.max(Z.shape),1)
        Z = Z - mean
        for i in range(0,np.max(Z.shape)):
            self.AR_prediction[i] = np.dot(self.calculate_AR_normal_matrix_x_row(Z, i, mean, std), self.AR_weights)
        
        self.AR_prediction = self.AR_prediction.transpose()[0] + mean
        return self.AR_prediction
    
    def get_previous_q_values(self,data,t):
        previous_q = np.zeros(self.q)
        j = 0
        for i in range(t-self.q,t):
            if i < 0:
                previous_q[j] = 0
            else:
                previous_q[j] = data[i]
            j+=1
        return previous_q
    
    def get_MA_prediction(self,data_set):
        self.MA_prediction = np.zeros(np.max(data_set.shape))
        Z = np.array(data_set)
        Z.shape = (np.max(Z.shape),1)
        for i in range(0,np.max(Z.shape)):
            self.MA_prediction[i] = np.average(self.get_previous_q_values(Z, i))
        
        return self.MA_prediction
    
    def calculate_AR_MA_normal_matrix_x_row(self,t):
        row = np.zeros((1,2))
        row[0][0] = self.MA_prediction[t]
        row[0][1] = self.AR_prediction[t]
        return row
    
    def calculate_AR_MA_weights(self):
        self.get_MA_prediction(self.training_data)
        self.get_AR_prediction(self.training_data)
        normal_matrix = np.zeros((len(self.training_data),2))
        
        for i in range(0,len(self.training_data)):
            normal_matrix[i] = self.calculate_AR_MA_normal_matrix_x_row(i)
        
        normal_matrix_tanspose = normal_matrix.transpose()
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(normal_matrix_tanspose,normal_matrix)),normal_matrix_tanspose),self.training_data)
        
#         print(self.weights)
#         #normalizing weigts
#         total = self.weights[0] + self.weights[1]
#         self.weights[0] = self.weights[0]/total
#         self.weights[1] = self.weights[1]/total
#         print(self.weights)
        
    def get_prediction(self, data_set):
        self.calculate_AR_MA_weights()
        
        self.get_MA_prediction(data_set)
        self.get_AR_prediction(data_set)
        Z = np.array(data_set)
        Z.shape = (np.max(Z.shape),1)
        self.prediction = np.zeros((np.max(Z.shape),1))
        for i in range(0,np.max(Z.shape)):
            self.prediction[i] = np.dot(self.calculate_AR_MA_normal_matrix_x_row(i), self.weights)
        
        self.prediction = self.prediction.transpose()[0]
        return self.prediction
        
    # Diagnostics and identification messures
    def mse(self,values,pridicted):
        error = 0.0
        for i in range(0,len(values)):
            error += (values[i] - pridicted[i])**2
        return error/len(values)
    
    def get_mse(self, data, prediction):
        return self.mse(data,prediction)
    
    def squared_error(self, data_set, prediction):
        x = np.arange(len(data_set))
        residuals = data_set - prediction 
        squared_error = residuals**2
        # mean = np.ones(len(data_set))*np.mean(residual)

        print("Standard Deviation of Residuals : " + str(np.std(residuals, ddof=1)))
        print("Mean of Residuals : " + str(np.mean(residuals)))
        return squared_error

    def find_anomalies(self, squared_errors):
        threshold = np.mean(squared_errors) + np.std(squared_errors)
        predictions = (squared_errors >= threshold).astype(int)
        return predictions, threshold
        
def detect_anamoly(df_result):
    time = df_result[list(df_result)[0]]
    values = df_result[list(df_result)[4]]
    values1 = df_result[list(df_result)[5]]
    print(values)
    print(values1)
    values2 = df_result[list(df_result)[6]]
    print(values2)
    training_ratio = 0.5
    validation_ratio = 0.5

    # Deviding the data set to training, validation, testing parts
    training_end = int(math.floor(len(values)*training_ratio))
    training_set_values = np.array(values[0:training_end])
    training_set_time = np.array(time[0:training_end])

    validation_start = training_end + 1
    validation_end = validation_start + int(math.floor(len(values)*(1-training_ratio)*validation_ratio))
    validation_set_values = np.array(values[validation_start:validation_end])
    validation_set_time = np.array(time[validation_start:validation_end])

    training_set_values2 = np.array(values2[0:training_end])
    training_set_time2 = np.array(time[0:training_end])

    validation_set_values2 = np.array(values2[validation_start:validation_end])
    validation_set_time2 = np.array(time[validation_start:validation_end])

    training_set_values1 = np.array(values1[0:training_end])
    training_set_time1 = np.array(time[0:training_end])

    validation_set_values1 = np.array(values1[validation_start:validation_end])
    validation_set_time1 = np.array(time[validation_start:validation_end])

    arma_model = ARMA(1,1)
    arma_model.set_training_data_set(training_set_values)
    arma_model.set_training_data_time(training_set_time)

    arma_model.set_validation_data_set(validation_set_values)
    arma_model.set_validation_data_time(validation_set_time)

    arma_model1 = ARMA(1,1)
    arma_model1.set_training_data_set(training_set_values1)
    arma_model1.set_training_data_time(training_set_time1)

    arma_model1.set_validation_data_set(validation_set_values1)
    arma_model1.set_validation_data_time(validation_set_time1)

    arma_model2 = ARMA(1,1)
    arma_model2.set_training_data_set(training_set_values2)
    arma_model2.set_training_data_time(training_set_time2)

    arma_model2.set_validation_data_set(validation_set_values2)
    arma_model2.set_validation_data_time(validation_set_time2)

    epochs = 10
    mse = np.zeros((epochs-1,epochs-1))

    for i in range(1, epochs):
        arma_model.set_p(i)
        for j in range(1,epochs):
            arma_model.set_q(j)

            prediction = arma_model.get_prediction(arma_model.validation_data_set)
            mse[i-1][j-1] = arma_model.get_mse(arma_model.validation_data_set, prediction)

            print("arma("+str(i)+","+str(j)+")", end=',')

    min_i = 0
    min_j = 0
    minimum = mse[0][0]
    for i in range(0,mse.shape[0]):
        for j in range(0, mse.shape[1]):
            if mse[i][j] < minimum:
                min_i = i
                min_j = j
                minimum = mse[i][j]

    print("MSE is minimum at P = "+str(min_i+1)+" and Q = "+str(min_j+1))

    arma_model = ARMA(min_i+1,min_j+1)
    whole_data = int(math.floor(len(values)))
    whole_data_set_values = np.array(values[0:whole_data])
    whole_data_set_time = np.array(time[0:whole_data])
    whole_data_set_values1 = np.array(values1[0:whole_data])
    whole_data_set_values2 = np.array(values2[0:whole_data])

    arma_model.set_training_data_set(training_set_values)
    arma_model.set_training_data_time(training_set_time)

    arma_model.set_testing_data_set(whole_data_set_values)
    arma_model.set_testing_data_time(whole_data_set_time)

    arma_model.set_validation_data_set(validation_set_values)
    arma_model.set_validation_data_time(validation_set_time)

    for i in range(1, epochs):
        arma_model1.set_p(i)
        for j in range(1,epochs):
            arma_model1.set_q(j)

            prediction = arma_model1.get_prediction(arma_model1.validation_data_set)
            mse[i-1][j-1] = arma_model1.get_mse(arma_model1.validation_data_set, prediction)

            print("arma("+str(i)+","+str(j)+")", end=',')

    min_i = 0
    min_j = 0
    minimum = mse[0][0]
    for i in range(0,mse.shape[0]):
        for j in range(0, mse.shape[1]):
            if mse[i][j] < minimum:
                min_i = i
                min_j = j
                minimum = mse[i][j]

    print("MSE is minimum at P = "+str(min_i+1)+" and Q = "+str(min_j+1))

    arma_model1 = ARMA(min_i+1,min_j+1)
    arma_model1.set_training_data_set(training_set_values1)
    arma_model1.set_training_data_time(training_set_time1)

    arma_model1.set_testing_data_set(whole_data_set_values1)
    arma_model1.set_testing_data_time(whole_data_set_time)

    arma_model1.set_validation_data_set(validation_set_values1)
    arma_model1.set_validation_data_time(validation_set_time1)

    for i in range(1, epochs):
        arma_model2.set_p(i)
        for j in range(1,epochs):
            arma_model2.set_q(j)

            prediction = arma_model2.get_prediction(arma_model2.validation_data_set)
            mse[i-1][j-1] = arma_model2.get_mse(arma_model2.validation_data_set, prediction)

            print("arma("+str(i)+","+str(j)+")", end=',')

    min_i = 0
    min_j = 0
    minimum = mse[0][0]
    for i in range(0,mse.shape[0]):
        for j in range(0, mse.shape[1]):
            if mse[i][j] < minimum:
                min_i = i
                min_j = j
                minimum = mse[i][j]

    print("MSE is minimum at P = "+str(min_i+1)+" and Q = "+str(min_j+1))

    arma_model2 = ARMA(min_i+1,min_j+1)
    arma_model2.set_training_data_set(training_set_values2)
    arma_model2.set_training_data_time(training_set_time2)

    arma_model2.set_testing_data_set(whole_data_set_values2)
    arma_model2.set_testing_data_time(whole_data_set_time)

    arma_model2.set_validation_data_set(validation_set_values2)
    arma_model2.set_validation_data_time(validation_set_time2)

    prediction = arma_model.get_prediction(arma_model.testing_data_set)
    squared_error = arma_model.squared_error(arma_model.testing_data_set, prediction)

    detects, thredhold = arma_model.find_anomalies(squared_error)

    prediction1 = arma_model1.get_prediction(arma_model1.testing_data_set)
    squared_error1 = arma_model1.squared_error(arma_model1.testing_data_set, prediction1)

    detects1, thredhold1 = arma_model1.find_anomalies(squared_error1)

    prediction2 = arma_model2.get_prediction(arma_model2.testing_data_set)
    squared_error2 = arma_model2.squared_error(arma_model2.testing_data_set, prediction2)

    detects2, thredhold2 = arma_model2.find_anomalies(squared_error2)

    df_result['predict'] = detects
    df_result['predict1'] = detects1
    df_result['predict2'] = detects2

    df_result.loc[df_result["predict"]==1, "CO2"] = float('NaN')
    df_result.loc[df_result["predict1"]==1, "CO"] = float('NaN')
    df_result.loc[df_result["predict2"]==1, "PM2.5"] = float('NaN')

    print(df_result.isnull().sum())
    return df_result


