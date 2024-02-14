#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Name: main.py
Description: This script fit a few machine learning models using data of asv abundance and environmental variables to predict
time-series asv abundance
Author: Zhifeng Yang
Created: 10/3/2023
Last Modified: 10/5/2023
Python Version: 3.9
"""

# Import necessary modules/packages
import configparser
import os
import pandas as pd
import numpy as np

## Sklearn Models 
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
## keras models
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout
from sklearn.preprocessing import QuantileTransformer
## reading config files

def read_config():
    """
    usage:
        read config files
    """
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    models_str=config['model_setting']['models']
    models=[key.strip() for key in models_str.split(',')]
    MLP_layer_str=config['MLP']['hidden_layer_sizes']
    MLP_layer=[int(key.strip()) for key in MLP_layer_str.split(',')]
    return {
        'dir_env': config['Paths']['dir_env'],
        'dir_asv': config['Paths']['dir_asv'],
        'dir_output':config['Paths']['dir_output'],
        'map_file':config['Paths']['map_file'],
        'predictor':config['model_setting']['predictor'],
        'models':models,
        'env_num':config['model_setting']['env_num'],
        'time_steps':config['model_setting']['time_steps'],
        'for_periods':config['model_setting']['for_periods'],
        'Ridge':{'alpha':float(config['Ridge']['alpha'])},
        'PLS':{'n_components':int(config['PLS']['n_components'])},
        'PCR':{'n_components':'None' if config['PCR']['n_components']=="None" else int(config['PCR']['n_components'])},
        'RandomForest':{'n_estimators':int(config['RandomForest']['n_estimators']),
                        'max_features':float(config['RandomForest']['max_features'])},
        'MLP':{'hidden_layer_sizes':MLP_layer,
               'alpha':float(config['MLP']['alpha']),
               'learning_rate_init':float(config['MLP']['learning_rate_init']),
               'max_iter':int(config['MLP']['max_iter'])},
        'DNN':{'n_layer':int(config['DNN']['n_layer']),
               'n_unit':int(config['DNN']['n_unit']),
               'dropout':float(config['DNN']['dropout'])},
        'RNN':{'n_layer':int(config['RNN']['n_layer']),
               'n_unit':int(config['RNN']['n_unit']),
               'dropout':float(config['RNN']['dropout'])},
        'LSTM':{'n_layer':int(config['LSTM']['n_layer']),
               'n_unit':int(config['LSTM']['n_unit']),
               'dropout':float(config['LSTM']['dropout'])}        
        }
    
import csv

def read_map_file(config):
    """
    usage:
        read map file that includes the file names of asv and environmental tables
    """
    file_path=config['map_file']
    env_files=[]
    asv_files=[]
    train_start=[]
    train_end=[]
    test_start=[]
    test_end=[]
    forecast_start=[]
    forecast_end=[]

    with open(file_path, 'r', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            env_files.append(row[0])
            asv_files.append(row[1])
            train_start.append(row[2])
            train_end.append(row[3])
            test_start.append(row[4])
            test_end.append(row[5])
            forecast_start.append(row[6])
            forecast_end.append(row[7])
    train_start=np.array(train_start[1:]).astype(int)
    train_end=np.array(train_end[1:]).astype(int)
    test_start=np.array(test_start[1:]).astype(int)
    test_end=np.array(test_end[1:]).astype(int)
    forecast_start=np.array(forecast_start[1:]).astype(int)
    forecast_end=np.array(forecast_end[1:]).astype(int)
    
    return env_files[1:],asv_files[1:],train_start,train_end,test_start,test_end,forecast_start,forecast_end

## read data, read one asv and one env
def read_data(env_files, asv_files,config):
    """
    usage：
        read asv file and env file
    
    Args:
        dir_env: directory of the environmental data files
        dir_asv: directory of the ASV data files
    """
    
    dir_env,dir_asv=config['dir_env'],config['dir_asv']
    os.chdir(dir_env)  # set dir location
    env_list=[]
    for env_file in env_files:
        env=pd.read_csv(env_file)
        env=env[env.columns[1:]].to_numpy(dtype='float32')
        env_list.append(env)
    
    
    os.chdir(dir_asv)  # set dir location
    asv_list=[]
    asvid_list=[]
    for asv_file in asv_files:
        asv=pd.read_csv(asv_file)
        if 'ASVID' in asv.columns:
            asvid=asv['ASVID']
        if 'ID' in asv.columns:
            asvid=asv['ID']
        asv=asv[asv.columns[1:]].to_numpy(dtype='float32').T
        asv_list.append(asv)
        asvid_list.append(asvid)

    return env_list, asv_list,asvid_list


## split data into training and test data
def split_samples(data,train_size,test_size):
    """ split samples for test (later) and train (former) samples"""
    train=data[(len(data)-test_size-train_size):(len(data)-test_size),:]
    test=data[(len(data)-test_size):,:]
    return train,test

## split data for time series data that use previous time points as predictor
def split_sample_ts(all_data,train_start,train_end,test_start,test_end,time_steps,for_periods,reshape=True):
    '''
    usage:
        split samples for predicting asv abundance of next a few time points
    
    args: 
        all_data: asv data
        test_end: end of test data
        train_size: training data size
        test_size: test data size
        time_steps: the number of previous time points as the predictor
        for_periods: the number of later time points as the predicted
        reshape: for rnn and lstm models, the input data requirs a 3-dimention data and needs no reshape (reshape=False)
      
    return:
        X_train: each row contains preious time_steps asv abundance of all asv abundance  for training data
        y_train: each row contains asv abundance of "for_periods" time points for training data
        X_test: X of test data set, same format as X_train
        y_test: y of test data set, same format as y_test
      
    '''
    # create training and test set
    ts_train = all_data[(train_start-1):train_end,:]
    ts_test  = all_data[(test_start-1):test_end,:]
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # create training data of {train_size-time_steps-for_periods+1} samples and {time_steps} time steps for X_train
    X_train = []
    y_train = []
    for i in range(time_steps,ts_train_len-for_periods+1): 
        X_train.append(ts_train[i-time_steps:i,:])
        y_train.append(ts_train[i+for_periods-1,:])
    if reshape:
        X_train, y_train = np.array(X_train).reshape(train_end-train_start+1-time_steps-for_periods+1,-1), np.array(y_train).reshape(train_end-train_start+1-time_steps-for_periods+1,-1)
    else:
        X_train, y_train = np.array(X_train), np.array(y_train)
  

    inputs = all_data[:,:]
    inputs = inputs[test_end-len(ts_test)-time_steps-for_periods+1:test_end]

    # Preparing X_test
    X_test = []
    y_test=[]
    for i in range(time_steps,ts_test_len+time_steps):
        X_test.append(inputs[i-time_steps:i,:])
        y_test.append(inputs[i+for_periods-1,:])
    if reshape:    
        X_test = np.array(X_test).reshape(test_end-test_start+1,-1)
        y_test=np.array(y_test).reshape(test_end-test_start+1,-1)
    else:
        X_test = np.array(X_test)
        y_test=np.array(y_test) 

    return X_train, y_train , X_test, y_test




## build models and associated data
## now only include models and data, may include optimizer later--zhifeng

def build_models_and_split_data(env_list,asv_list,train_start,train_end,test_start,test_end,forecast_start,forecast_end,config):
    """
    usage:
        build models and split data into training and test dataset;
        for each model, prepare the appropriate form of data
    
    args: 
        model_names: names of models in config file
        predictor: 
            predictor types, currently, could be "env" (current environmental factors as X) 
            or "env+asv" (current environmental variables and previous asv abundance as X)
        env_list,asv_list: list of env and asv tables for different data sets
        test_end,train_size,test_size,env_num,time_steps,for_periods: for spliting the training data and test data, and
    
    return:
        a dictionary contains models and datasets for all models in the format of {'model':{'model': defined_model, 'data':
        [[X_train,X_test,y_train,y_test] for all data sets]}}
    
    """
    data_model={}   ## need define a class for the structure (includes data, model, optimizer, etc.) later--zhifeng
    data_list=[]
    model_names=config['models']
    predictor=config['predictor']   
    env_num=config['env_num']  
    time_steps=int(config['time_steps'])  
    for_periods=int(config['for_periods'])  
    
    ## split data based on the selection of predictors
    # if predictor == "env":   ## use previous environmental variables to predict asv abundance
    #     for env, asv,train_start_i,train_end_i,test_start_i,test_end_i in zip(env_list,asv_list,train_start,train_end,test_start,test_end):
    #         ## split asv abundance as X and Y
    #         X_train, y_train , X_test, y_test=split_sample_ts(asv,train_start_i,train_end_i,test_start_i,test_end_i,time_steps,for_periods)
    #         ## split environmental variables as X
    #         X_train2,X_test2=split_sample_ts(env[:,:env_num],train_start_i,train_end_i,test_start_i,test_end_i,time_steps,for_periods)
    #         ## For forcast, use last {time_steps} of avaiable data for X
    #         X_last=env[-time_steps:,:env_num].reshape(1,-1)
            
    #         X_train=X_train2
    #         X_test=X_test2[0:1,:]
    #         y_train=y_train
    #         y_test=y_test
    #         X=asv
    #         data_list.append([X_train,X_test,y_train,y_test,X_last,X])
    
    if predictor == "asv":  ## use previous asv abundance to predict asv abundance
        for asv,train_start_i,train_end_i,test_start_i,test_end_i, forecast_start_i,forecast_end_i in zip(asv_list,train_start,train_end,test_start,test_end,forecast_start,forecast_end):
            ## split asv abundance as X and Y
            X_train, y_train , X_test, y_test=split_sample_ts(asv,train_start_i,train_end_i,test_start_i,test_end_i,time_steps,for_periods)
            X_test=X_test[0:1,:]
            if forecast_start_i>asv.shape[0]:
                X_last=asv[-time_steps:,:].reshape(1,-1)
            else:
                X_last=asv[:(forecast_start_i-1),:]
                X_last=X_last[-time_steps:,:].reshape(1,-1)
            X=asv
            data_list.append([X_train,X_test,y_train,y_test,X_last,X])
    
    if predictor == "env+asv":  ## use previous asv abundance and previous environmental variables to predict asv abundance
        for env, asv,train_start_i,train_end_i,test_start_i,test_end_i, forecast_start_i,forecast_end_i in zip(env_list,asv_list,train_start,train_end,test_start,test_end,forecast_start,forecast_end):
            env_num=config['env_num']  
            if env_num=="all":
                env_num=env.shape[1]
            elif env_num>env.shape[1]:
                env_num=env.shape[1]  
                print("The number of enviromental factors is larger than all the number, so use all the environmental factors as X")
            ## split asv abundance as X and Y
            X_train, y_train , X_test, y_test=split_sample_ts(np.concatenate((asv,env[:,:env_num]),axis=1),train_start_i,train_end_i,test_start_i,test_end_i,time_steps,for_periods)
            X_test=X_test[0:1,:]
            if forecast_start_i>asv.shape[0]:
                X_last=np.concatenate((asv,env[:,:env_num]),axis=1)[-time_steps:,:].reshape(1,-1)
            else:
                X_last=np.concatenate((asv,env[:,:env_num]),axis=1)[:(forecast_start_i-1),:]
                X_last=X_last[-time_steps:,:].reshape(1,-1)                
            X=asv
            data_list.append([X_train,X_test,y_train,y_test,X_last,X])

            
    ## define models and stored it with its datasets
    if "Dummy" in model_names:
        dummy=make_pipeline(DummyRegressor(strategy="mean"))
        data_model['Dummy']={'model':dummy,'data':data_list}

    if 'LinearRegression' in model_names:    
        lr=make_pipeline(StandardScaler(),LinearRegression())
        data_model['LinearRegression']={'model':lr,'data':data_list}
    if 'Ridge' in model_names:
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=config['Ridge']['alpha']))
        data_model['Ridge']={'model':ridge,'data':data_list}    
    if 'PLS' in model_names:
        pls=make_pipeline(StandardScaler(),PLSRegression(config['PLS']['n_components']))
        data_model['PLS']={'model':pls,'data':data_list} 
    if 'PCR' in model_names:
        pcr = make_pipeline(StandardScaler(), PCA(n_components=config['PCR']['n_components']),LinearRegression())
        data_model['PCR']={'model':pcr,'data':data_list}
    if 'RandomForest' in model_names:
        rf=make_pipeline(StandardScaler(),RandomForestRegressor(random_state=0,n_estimators=config['RandomForest']['n_estimators'],max_features=config['RandomForest']['max_features']))
        data_model['RandomForest']={'model':rf,'data':data_list}
    if 'MLP' in model_names: ## It looks like standardscaler will produce some bugs for MLP
        mlp=make_pipeline(QuantileTransformer(),MLPRegressor(random_state=1,activation="relu", learning_rate="adaptive", learning_rate_init=config['MLP']['learning_rate_init'], max_iter=config['MLP']['max_iter'],alpha=config['MLP']['alpha'],hidden_layer_sizes=config['MLP']['hidden_layer_sizes']))
        data_model['MLP']={'model':mlp,'data':data_list}     
    ## For different datasets, we need build different keras models as the input dimension is dependent on the dimension of X
    if 'DNN' in model_names:
        data_model['DNN']={'model':[],'data':data_list} 
        for data in data_model['DNN']['data']:
            input_dimension = data[0].shape[1] # Replace with the appropriate input dimension
            output_dimension=data[2].shape[1]
            def create_dnn_regressor(input_dimension=input_dimension,output_dimension=output_dimension,n_layer=config['DNN']['n_layer'],n_unit=config['DNN']['n_unit'],dropout=config['DNN']['dropout']):
                model = Sequential()
                model.add(Dense(units=n_unit, activation='relu', input_dim=input_dimension))
                model.add(Dropout(dropout))
                for i in range(n_layer-1):
                    model.add(Dense(units=n_unit, activation='relu'))
                    model.add(Dropout(dropout))# Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
                return model
            model = KerasRegressor(build_fn=create_dnn_regressor, epochs=500, batch_size=32, verbose=0)
            data_model['DNN']['model'].append(model)
     
    ## For RNN and LSTM models, only previous time points of asv abundance can be used as predictor
    if 'RNN' in model_names:
        data_model['RNN']={'model':[],'data':[]} 
        for asv,train_start_i,train_end_i,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(asv_list,train_start,train_end,test_start,test_end,forecast_start,forecast_end):
            X_train, y_train , X_test, y_test=split_sample_ts(asv,train_start_i,train_end_i,test_start_i,test_end_i,time_steps,for_periods,reshape=False)
            X_test=X_test[0:1,:]
            if forecast_start_i>asv.shape[0]:
                X_last=asv[-time_steps:,:].reshape(1,time_steps,-1)
            else:
                X_last=asv[:(forecast_start_i-1),:]
                X_last=X_last[-time_steps:,:].reshape(1,time_steps,-1)
            X=asv
            data_model['RNN']['data'].append([X_train,X_test,y_train,y_test,X_last,X])
            # Specify the input dimension (number of features) and sequence length
            input_dimension = X_train.shape[2]# Replace with the appropriate input dimension
            sequence_length = X_train.shape[1]# Replace with the appropriate sequence length
            output_dimension=y_train.shape[1]
            def create_rnn_regressor(input_dimension=input_dimension,sequence_length=sequence_length,output_dimension=output_dimension,n_layer=config['RNN']['n_layer'],n_unit=config['RNN']['n_unit'],dropout=config['RNN']['dropout']):
                from keras.layers import SimpleRNN
                model = Sequential()
                model.add(SimpleRNN(units=n_unit, activation='relu', input_shape=(sequence_length, input_dimension)))
                model.add(Dropout(dropout))
                for i in range(n_layer-1):
                    model.add(Dense(units=n_unit, activation='relu'))
                    model.add(Dropout(dropout))# Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
                return model


            # Create a KerasRegressor instance using your RNN regressor model function
            model = KerasRegressor(build_fn=create_rnn_regressor, 
                                   epochs=200, batch_size=32,verbose=0)
            data_model['RNN']['model'].append(model)
    if 'LSTM' in model_names:
        data_model['LSTM']={'model':[],'data':[]} 
        for asv,train_start_i,train_end_i,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(asv_list,train_start,train_end,test_start,test_end,forecast_start,forecast_end):
            X_train, y_train , X_test, y_test=split_sample_ts(asv,train_start_i,train_end_i,test_start_i,test_end_i,time_steps,for_periods,reshape=False)
            X_test=X_test[0:1,:]
            if forecast_start_i>asv.shape[0]:
                X_last=asv[-time_steps:,:].reshape(1,time_steps,-1)
            else:
                X_last=asv[:(forecast_start_i-1),:]
                X_last=X_last[-time_steps:,:].reshape(1,time_steps,-1)
            X=asv
            data_model['LSTM']['data'].append([X_train,X_test,y_train,y_test,X_last,X])
            # Specify the input dimension (number of features) and sequence length
            input_dimension = X_train.shape[2]# Replace with the appropriate input dimension
            sequence_length = X_train.shape[1]# Replace with the appropriate sequence length
            output_dimension=y_train.shape[1]
            # Create a KerasRegressor instance using your RNN regressor model function
            def create_lstm_regressor(input_dimension=input_dimension,sequence_length=sequence_length,
                                      output_dimension=output_dimension,n_layer=config['LSTM']['n_layer'],n_unit=config['LSTM']['n_unit'],dropout=config['LSTM']['dropout']):
                from keras.layers import LSTM
                model = Sequential()
                model.add(LSTM(units=n_unit, activation='relu', input_shape=(sequence_length, input_dimension)))
                model.add(Dropout(dropout))
                for i in range(n_layer-1):
                    model.add(Dense(units=n_unit, activation='relu'))
                    model.add(Dropout(dropout))# Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
                return model

            model = KerasRegressor(build_fn=create_lstm_regressor, epochs=200, batch_size=32,verbose=0)
            data_model['LSTM']['model'].append(model)
    return data_model



## model fit and predict
def models_fit_predict(data_model,asvid_list,test_start,test_end,forecast_start,forecast_end, config):
    """
    usage:
        Fit a list of scikit-learn and keras models with training data and predict asv abundance for test data
    -------
    return:
    generate predicted asv table in output directory

    """
    dir_output=config['dir_output']
    predictor=config['predictor']
    time_steps=int(config['time_steps'])
    
    os.chdir(dir_output)  # set dir location
    for model_name, model_data in data_model.items():
        
        
        ## For each model, each data, fit the model, make prediction for test data and make prediction for future data using the last {time_steps} of data as predictor
        ## The prediction is iterative.
        if model_name in ["DNN"]:
            n=0
            for model,data,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['model'],
                                                                                          model_data['data'],test_start,test_end,forecast_start,forecast_end):         
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]
                model.fit(X_train, y_train)
                num_iterations = y_test.shape[0]
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    # Predict the next time point
                    print("X_test[:,1:10] for "+str(i)+"th iteration")
                    print(X_test[:,1:10])
                    next_prediction = model.predict(X_test)
                    next_time=int(X_test.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_test = np.concatenate((X_test[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                df = pd.DataFrame(y_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(test_start_i,test_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.predicted.asv.csv')
                
                if forecast_start_i>X.shape[0]:
                    num_iterations = forecast_end_i-X.shape[0]
                else:
                    num_iterations=forecast_end_i-forecast_start_i+1
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    # Predict the next time point
                    next_prediction = model.predict(X_last)
                    next_time=int(X_last.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_last = np.concatenate((X_last[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                if forecast_start_i>X.shape[0]:
                    df = pd.DataFrame(y_pred[(forecast_start_i-X.shape[0]-1):(forecast_end_i-X.shape[0]),:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                else:
                    df = pd.DataFrame(y_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))

                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.forecasted.asv.csv')                
    
                n+=1
                
        elif model_name in ["RNN","LSTM"]:
            n=0
            for model,data,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['model'],
                                                                                          model_data['data'],test_start,test_end,forecast_start,forecast_end):         
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]
                model.fit(X_train, y_train)
                num_iterations = y_test.shape[0]
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    print("X_test[:,:,0] for "+str(i)+"th iteration")
                    print(X_test[:,:,0])
                    # Predict the next time point
                    next_prediction = model.predict(X_test)
                    next_time=int(X_test.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_test = np.concatenate((X_test[:,next_time:,:], next_prediction.reshape(1,1,-1)),axis=1)

                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                df = pd.DataFrame(y_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(test_start_i,test_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.predicted.asv.csv')
                
                if forecast_start_i>X.shape[0]:
                    num_iterations = forecast_end_i-X.shape[0]
                else:
                    num_iterations=forecast_end_i-forecast_start_i+1
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    # Predict the next time point
                    next_prediction = model.predict(X_last)
                    next_time=int(X_last.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_last = np.concatenate((X_last[:,next_time:], next_prediction.reshape(1,1,-1)),axis=1)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                if forecast_start_i>X.shape[0]:
                    df = pd.DataFrame(y_pred[(forecast_start_i-X.shape[0]-1):(forecast_end_i-X.shape[0]),:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                else:
                    df = pd.DataFrame(y_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.forecasted.asv.csv')                
                n+=1
        ## For other sklearn models
        else:
            n=0
            model=model_data['model']
            for data,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['data'],test_start,test_end,forecast_start,forecast_end):         
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]
                model.fit(X_train, y_train)
                num_iterations = y_test.shape[0]
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    print("X_test[:,1:10] for "+str(i)+"th iteration")
                    print(X_test[:,1:10])
                    # Predict the next time point
                    next_prediction = model.predict(X_test)
                    next_time=int(X_test.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_test = np.concatenate((X_test[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)    
                df = pd.DataFrame(y_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(test_start_i,test_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.predicted.asv.csv')
                
                if forecast_start_i>X.shape[0]:
                    num_iterations = forecast_end_i-X.shape[0]
                else:
                    num_iterations=forecast_end_i-forecast_start_i+1
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    # Predict the next time point
                    next_prediction = model.predict(X_last)
                    next_time=int(X_last.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_last = np.concatenate((X_last[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                if forecast_start_i>X.shape[0]:
                    df = pd.DataFrame(y_pred[(forecast_start_i-X.shape[0]-1):(forecast_end_i-X.shape[0]),:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                else:
                    df = pd.DataFrame(y_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))

                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.forecasted.asv.csv')                
                n+=1 




if __name__ == "__main__":
    config = read_config()
    
    print("Model configuration:")
    print(f"Models: {config['models']}")
    print(f"Predictors: {config['predictor']}")
    print(f"env_num: {config['env_num']}")
    print(f"time_steps: {config['time_steps']}")
    print(f"for_periods: {config['for_periods']}")
    
    print("---------------------")    
    print("reading map file...")
    env_files,asv_files,train_start,train_end,test_start,test_end,forecast_start,forecast_end=read_map_file(config)
    print("reading data...")
    env_list, asv_list,asvid_list=read_data(env_files,asv_files,config)
    print("building models...")
    data_model=build_models_and_split_data(env_list,asv_list,train_start,train_end,test_start,test_end,forecast_start,forecast_end,config)
    print("model fitting and predicting...")
    models_fit_predict(data_model,asvid_list,test_start,test_end, forecast_start,forecast_end, config)
    
    print("---------------------")
    print(f"predicted asv tables for test and future data points are stored in {config['dir_output']}")


    
    

