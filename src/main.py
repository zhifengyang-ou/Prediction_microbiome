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

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
## keras models
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import QuantileTransformer
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, GRU, Input,MultiHeadAttention, LayerNormalization
from keras.models import Model

## ARIMA model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR


## pytorch models
#from torch.utils.data import DataLoader, TensorDataset
#import torch
#import pytorch_module
## evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        'GradientBoostingRegressor': {
            'n_estimators': int(config['GradientBoostingRegressor']['n_estimators']),
            'learning_rate': float(config['GradientBoostingRegressor']['learning_rate']),
            'max_depth': int(config['GradientBoostingRegressor']['max_depth']),
            'subsample': float(config['GradientBoostingRegressor']['subsample'])
        },
        'SVR': {
            'C': float(config['SVR']['C']),
            'epsilon': float(config['SVR']['epsilon']),
            'kernel': config['SVR']['kernel']
        },       
        'DNN':{'n_layer':int(config['DNN']['n_layer']),
               'epoch':int(config['DNN']['epoch']),
               'n_unit':int(config['DNN']['n_unit']),
               'dropout':float(config['DNN']['dropout']),
               'learning_rate':float(config['pytorch_DNN']['learning_rate'])},
               
        'RNN':{'n_layer':int(config['RNN']['n_layer']),
               'n_unit':int(config['RNN']['n_unit']),
               'epoch':int(config['RNN']['epoch']),
               'dropout':float(config['RNN']['dropout']),
               'learning_rate':float(config['pytorch_RNN']['learning_rate'])},
               
        'LSTM':{'n_layer':int(config['LSTM']['n_layer']),
               'n_unit':int(config['LSTM']['n_unit']),
               'epoch':int(config['LSTM']['epoch']),
               'dropout':float(config['LSTM']['dropout']),
               'learning_rate':float(config['pytorch_LSTM']['learning_rate'])},
        'GRU':{'n_layer' : int(config['GRU']['n_layer']),
                'n_unit' : int(config['GRU']['n_unit']),
               'dropout' : float(config['GRU']['dropout']),
                'learning_rate' : float(config['GRU']['learning_rate']),
                'epoch' : int(config['GRU']['epoch'])
               },
        'Transformer':{'n_heads' : int(config['Transformer']['n_heads']),
                        'n_units' : int(config['Transformer']['n_units']),
                        'n_layers' : int(config['Transformer']['n_layers']),
                        'dropout' : float(config['Transformer']['dropout']),
                        'learning_rate' : float(config['Transformer']['learning_rate']),
                        'epoch': int(config['Transformer']['epoch'])
               },
               
        'pytorch_DNN':{'n_layer':int(config['pytorch_DNN']['n_layer']),
               'n_unit':int(config['pytorch_DNN']['n_unit']),
               'dropout':float(config['pytorch_DNN']['dropout']),
               'epoch':int(config['pytorch_DNN']['epoch']),
               'learning_rate':float(config['pytorch_DNN']['learning_rate'])},
               
        'pytorch_RNN':{'n_layer':int(config['pytorch_RNN']['n_layer']),
               'n_unit':int(config['pytorch_RNN']['n_unit']),
               'dropout':float(config['pytorch_RNN']['dropout']),
               'epoch':int(config['pytorch_RNN']['epoch']),
               'learning_rate':float(config['pytorch_RNN']['learning_rate'])},
               
        'pytorch_LSTM':{'n_layer':int(config['pytorch_LSTM']['n_layer']),
               'n_unit':int(config['pytorch_LSTM']['n_unit']),
               'dropout':float(config['pytorch_LSTM']['dropout']),
               'epoch':int(config['pytorch_LSTM']['epoch']),
               'learning_rate':float(config['pytorch_LSTM']['learning_rate'])},
        'ARIMA':{'p':int(config['model_setting']['time_steps']),
               'd':int(config['ARIMA']['d']),
               'q':float(config['ARIMA']['q'])}               
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
        
        data_model['PLS']={'model':[],'data':data_list}
        for data in data_model['PLS']['data']:
            n_components=config['PLS']['n_components']
            if config['PLS']['n_components']>data[1].shape[1]:
                n_components=  data[1].shape[1]
            pls=make_pipeline(StandardScaler(),PLSRegression(n_components))
            data_model['PLS']['model'].append(pls)
    if 'PCR' in model_names:
        data_model['PCR']={'model':[],'data':data_list}
        for data in data_model['PCR']['data']:
            n_components=min(config['PCR']['n_components'],data[0].shape[0],data[0].shape[1])    
            pcr = make_pipeline(StandardScaler(), PCA(n_components=n_components),LinearRegression())
            data_model['PCR']['model'].append(pcr)
    if 'RandomForest' in model_names:
        rf=make_pipeline(StandardScaler(),RandomForestRegressor(random_state=0,n_estimators=config['RandomForest']['n_estimators'],max_features=config['RandomForest']['max_features']))
        data_model['RandomForest']={'model':rf,'data':data_list}
    if 'MLP' in model_names: ## It looks like standardscaler will produce some bugs for MLP
        mlp=make_pipeline(QuantileTransformer(),MLPRegressor(random_state=1,activation="relu", learning_rate="adaptive", learning_rate_init=config['MLP']['learning_rate_init'], max_iter=config['MLP']['max_iter'],alpha=config['MLP']['alpha'],hidden_layer_sizes=config['MLP']['hidden_layer_sizes']))
        data_model['MLP']={'model':mlp,'data':data_list}    
    if 'GradientBoostingRegressor' in model_names:
        gtb = make_pipeline(StandardScaler(),MultiOutputRegressor(GradientBoostingRegressor(random_state=0,n_estimators=config['GradientBoostingRegressor']['n_estimators'],   # More boosting stages
            learning_rate=config['GradientBoostingRegressor']['learning_rate'], 
            max_depth=config['GradientBoostingRegressor']['max_depth'],        
            subsample=config['GradientBoostingRegressor']['subsample'])))   
        data_model['GradientBoostingRegressor'] = {'model': gtb, 'data': data_list}

    if 'SVR' in model_names:
        svr = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=config['SVR']['C'],
            epsilon=config['SVR']['epsilon'],
            kernel=config['SVR']['kernel'])))
        data_model['SVR'] = {'model': svr, 'data': data_list}
    ## For different datasets, we need build different keras models as the input dimension is dependent on the dimension of X
    if 'DNN' in model_names:
        data_model['DNN']={'model':[],'data':data_list} 
        for data in data_model['DNN']['data']:
            input_dimension = data[0].shape[1] # Replace with the appropriate input dimension
            output_dimension=data[2].shape[1]
            def create_dnn_regressor(input_dimension=input_dimension,output_dimension=output_dimension,n_layer=config['DNN']['n_layer'],n_unit=config['DNN']['n_unit'],dropout=config['DNN']['dropout'],learning_rate=config['DNN']['learning_rate']):
                model = Sequential()
                model.add(Dense(units=n_unit, activation='relu', input_dim=input_dimension))
                model.add(Dropout(dropout))
                for i in range(n_layer-1):
                    model.add(Dense(units=n_unit, activation='relu'))
                    model.add(Dropout(dropout))# Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
                return model
            model = KerasRegressor(model=create_dnn_regressor, epochs=config['DNN']['epoch'], batch_size=32, verbose=0)
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
            def create_rnn_regressor(input_dimension=input_dimension,sequence_length=sequence_length,output_dimension=output_dimension,n_layer=config['RNN']['n_layer'],n_unit=config['RNN']['n_unit'],dropout=config['RNN']['dropout'],learning_rate=config['RNN']['learning_rate']):
                from keras.layers import SimpleRNN
                model = Sequential()
                model.add(SimpleRNN(units=n_unit, activation='relu', input_shape=(sequence_length, input_dimension)))
                model.add(Dropout(dropout))
                for i in range(n_layer-1):
                    model.add(Dense(units=n_unit, activation='relu'))
                    model.add(Dropout(dropout))# Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
                return model


            # Create a KerasRegressor instance using your RNN regressor model function
            model = KerasRegressor(model=create_rnn_regressor, 
                                   epochs=config['RNN']['epoch'], batch_size=32,verbose=0)
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
                                      output_dimension=output_dimension,n_layer=config['LSTM']['n_layer'],n_unit=config['LSTM']['n_unit'],dropout=config['LSTM']['dropout'],learning_rate=config['LSTM']['learning_rate']):
                from keras.layers import LSTM
                model = Sequential()
                model.add(LSTM(units=n_unit, activation='relu', input_shape=(sequence_length, input_dimension)))
                model.add(Dropout(dropout))
                for i in range(n_layer-1):
                    model.add(Dense(units=n_unit, activation='relu'))
                    model.add(Dropout(dropout))# Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
                return model

            model = KerasRegressor(model=create_lstm_regressor, epochs=config['LSTM']['epoch'], batch_size=32,verbose=0)
            data_model['LSTM']['model'].append(model)
    if 'GRU' in model_names:
        def create_gru_regressor(input_dimension, sequence_length, output_dimension, n_layer, n_unit, dropout, learning_rate):
            """ Create GRU model """
            model = Sequential()
            model.add(GRU(units=n_unit, activation='relu', input_shape=(sequence_length, input_dimension)))
            model.add(Dropout(dropout))
            for _ in range(n_layer - 1):
                model.add(Dense(units=n_unit, activation='relu'))
                model.add(Dropout(dropout))
            model.add(Dense(units=output_dimension, activation='linear'))
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
            return model
        data_model['GRU'] = {'model': [], 'data': []}
        for asv, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(
                asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end):
            X_train, y_train, X_test, y_test = split_sample_ts(asv, train_start_i, train_end_i, test_start_i, test_end_i, time_steps, for_periods, reshape=False)
            X_test=X_test[0:1,:]
            if forecast_start_i>asv.shape[0]:
                X_last=asv[-time_steps:,:].reshape(1,time_steps,-1)
            else:
                X_last=asv[:(forecast_start_i-1),:]
                X_last=X_last[-time_steps:,:].reshape(1,time_steps,-1)
            X=asv
            data_model['GRU']['data'].append([X_train,X_test,y_train,y_test,X_last,X])
            model = KerasRegressor(
                model=create_gru_regressor(input_dimension=X_train.shape[2], sequence_length=X_train.shape[1], output_dimension=y_train.shape[1],
                                           n_layer=config['GRU']['n_layer'], n_unit=config['GRU']['n_unit'], dropout=config['GRU']['dropout'], learning_rate=config['GRU']['learning_rate']),
                epochs=config['GRU']['epoch'], batch_size=32, verbose=0
            )
            data_model['GRU']['model'].append(model)
    if 'Transformer' in model_names:
        data_model['Transformer'] = {'model': [], 'data': []}
        def create_transformer_regressor(input_dimension, sequence_length, output_dimension, n_heads, n_units, n_layers, dropout, learning_rate):
            """ Create Transformer model """
            from keras.layers import GlobalAveragePooling1D,GlobalMaxPooling1D
            inputs = Input(shape=(sequence_length, input_dimension))
            x = inputs
            x=Dense(n_units, activation='linear')(x)
            for _ in range(n_layers):
                attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=n_units)(x, x)
                x = LayerNormalization()(x + attn_output)
                ffn_output = Dense(n_units, activation='relu')(x)
                x = LayerNormalization()(x + ffn_output)

            outputs = Dense(output_dimension, activation='linear')(x)
            outputs = GlobalAveragePooling1D()(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])
            return model
        for asv, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(
                asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end):
            X_train, y_train, X_test, y_test = split_sample_ts(asv, train_start_i, train_end_i, test_start_i, test_end_i, time_steps, for_periods, reshape=False)
            X_test=X_test[0:1,:]
            if forecast_start_i>asv.shape[0]:
                X_last=asv[-time_steps:,:].reshape(1,time_steps,-1)
            else:
                X_last=asv[:(forecast_start_i-1),:]
                X_last=X_last[-time_steps:,:].reshape(1,time_steps,-1)
            X=asv
            data_model['Transformer']['data'].append([X_train,X_test,y_train,y_test,X_last,X])
            
            
            model = KerasRegressor(
                model=create_transformer_regressor(input_dimension=X_train.shape[2], sequence_length=X_train.shape[1], output_dimension=y_train.shape[1],
                                                   n_heads=config['Transformer']['n_heads'], n_units=config['Transformer']['n_units'],
                                                   n_layers=config['Transformer']['n_layers'], dropout=config['Transformer']['dropout'], learning_rate=config['Transformer']['learning_rate']),
                epochs=config['Transformer']['epoch'], batch_size=32, verbose=0
            )
            data_model['Transformer']['model'].append(model)
            data_model['Transformer']['data'].append([X_train, X_test, y_train, y_test,asv])
    if 'ARIMA' in model_names:
        data_model['ARIMA'] = {'model': [], 'data': []}
        for asv, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end):
            X_train, y_train, X_test, y_test=asv[train_start_i-1:train_end_i,:],asv[train_start_i-1:train_end_i,:],asv[test_start_i-1:test_end_i,:],asv[test_start_i-1:test_end_i,:]
            # Since ARIMA takes a 1D series as input, we fit the model for each ASV time series separately
            arima_models = []
            for i in range(X_train.shape[1]):  # Iterate over ASV abundance columns
                arima_model = ARIMA(X_train[:, i], order=(config['ARIMA']['p'], config['ARIMA']['d'], config['ARIMA']['q']))  # Example ARIMA order (p, d, q)
                arima_fitted = arima_model.fit()
                arima_models.append(arima_fitted)
            
            data_model['ARIMA']['model'].append(arima_models)
            X=asv
            data_model['ARIMA']['data'].append([X_train, X_test, y_train, y_test,X])
    if 'VAR' in model_names:
        data_model['VAR'] = {'model': [], 'data': []}
        for env,asv, train_start_i, train_end_i, test_start_i, test_end_i in zip(env_list,asv_list, train_start, train_end, test_start, test_end):
            if predictor == "asv":
                X = asv
            else:
                X=np.concatenate((asv,env[:,:env_num]),axis=1)
            # Create training and test datasets by stacking all time series (ASV abundance)
            X_train = X[train_start_i - 1:train_end_i, :]
            X_test = X[test_start_i - 1:test_end_i, :]
            
                    
            # Fit the VAR model using all ASV time series as endogenous variables
            var_model = VAR(X_train)
            var_fitted = var_model.fit(maxlags=time_steps)  # Use lags parameter from config

            data_model['VAR']['model'].append(var_fitted)
            data_model['VAR']['data'].append([X_train, X_test,asv])

            
    if 'pytorch_DNN' in model_names:

        data_model = {'pytorch_DNN': {'model': [], 'data': data_list}}
        for data in data_model['pytorch_DNN']['data']:
            input_dimension = data[0].shape[1]  # Replace with the appropriate input dimension
            output_dimension = data[2].shape[1]
            n_layer = config['pytorch_DNN']['n_layer']
            n_unit = config['pytorch_DNN']['n_unit']
            dropout = config['pytorch_DNN']['dropout']
    
            model, optimizer, loss_fn = pytorch_module.create_dnn_regressor(input_dimension, output_dimension, n_layer=config['pytorch_DNN']['n_layer'],n_unit=config['pytorch_DNN']['n_unit'],dropout=config['pytorch_DNN']['dropout'],lr=config['pytorch_DNN']['learning_rate'])
            data_model['pytorch_DNN']['model'].append((model, optimizer, loss_fn))
    
    if 'pytorch_RNN' in model_names:
        data_model['pytorch_RNN'] = {'model': [], 'data': []}
        for asv, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end):
            X_train, y_train, X_test, y_test = split_sample_ts(asv, train_start_i, train_end_i, test_start_i, test_end_i, time_steps, for_periods, reshape=False)
            X_test = X_test[0:1, :]
            if forecast_start_i > asv.shape[0]:
                X_last = asv[-time_steps:, :].reshape(1, time_steps, -1)
            else:
                X_last = asv[:(forecast_start_i - 1), :]
                X_last = X_last[-time_steps:, :].reshape(1, time_steps, -1)
            X = asv
            data_model['pytorch_RNN']['data'].append([X_train, X_test, y_train, y_test, X_last, X])
            input_dimension = X_train.shape[2]
            sequence_length = X_train.shape[1]
            output_dimension = y_train.shape[1]
            model, optimizer, loss_fn = pytorch_module.create_pytorch_rnn_regressor(input_dimension, sequence_length, output_dimension, config['pytorch_RNN']['n_layer'], config['pytorch_RNN']['n_unit'], config['pytorch_RNN']['dropout'], config['pytorch_RNN']['learning_rate'])
            data_model['pytorch_RNN']['model'].append((model, optimizer, loss_fn))
    
    if 'pytorch_LSTM' in model_names:
        data_model['pytorch_LSTM'] = {'model': [], 'data': []}
        for asv, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end):
            X_train, y_train, X_test, y_test = split_sample_ts(asv, train_start_i, train_end_i, test_start_i, test_end_i, time_steps, for_periods, reshape=False)
            X_test = X_test[0:1, :]
            if forecast_start_i > asv.shape[0]:
                X_last = asv[-time_steps:, :].reshape(1, time_steps, -1)
            else:
                X_last = asv[:(forecast_start_i - 1), :]
                X_last = X_last[-time_steps:, :].reshape(1, time_steps, -1)
            X = asv
            data_model['pytorch_LSTM']['data'].append([X_train, X_test, y_train, y_test, X_last, X])
            input_dimension = X_train.shape[2]
            sequence_length = X_train.shape[1]
            output_dimension = y_train.shape[1]
            model, optimizer, loss_fn = pytorch_module.create_pytorch_lstm_regressor(input_dimension, sequence_length, output_dimension, config['pytorch_LSTM']['n_layer'], config['pytorch_LSTM']['n_unit'], config['pytorch_LSTM']['dropout'], config['pytorch_LSTM']['learning_rate'])
            data_model['pytorch_LSTM']['model'].append((model, optimizer, loss_fn))
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
    training_errors = {}
    test_errors={}
    for model_name, model_data in data_model.items():
        training_errors[model_name] = []  
        test_errors[model_name] = []
        ## For each model, each data, fit the model, make prediction for test data and make prediction for future data using the last {time_steps} of data as predictor
        ## The prediction is iterative.
        if model_name in ["DNN"]:
            n=0
            for model,data,train_start_i,train_end_i,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['model'],
                                                                                          model_data['data'],train_start,train_end,test_start,test_end,forecast_start,forecast_end):         
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]
                model.fit(X_train, y_train)
                #write training predicted
                y_train_pred=model.predict(X_train)
                df = pd.DataFrame(y_train_pred[:,:X.shape[1]], columns = asvid_list[n],index=range(train_start_i+time_steps,train_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.train_predicted.asv.csv') 
                ## model evaluation for training
                y_train_pred = model.predict(X_train)
                mse = mean_squared_error(y_train, y_train_pred)
                mae = mean_absolute_error(y_train, y_train_pred)
                rmse = np.sqrt(mse)
                training_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
                num_iterations = y_test.shape[0]
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    # Predict the next time point
                    # print("X_test[:,1:10] for "+str(i)+"th iteration")
                    # print(X_test[:,1:10])
                    next_prediction = model.predict(X_test)
                    next_time=int(X_test.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_test = np.concatenate((X_test[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)               

                ## model evaluation for test
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                test_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
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
                
        elif model_name in ["RNN","LSTM",'GRU','Transformer']:
            n=0
            for model,data,train_start_i,train_end_i,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['model'],
                                                                                          model_data['data'],train_start,train_end,test_start,test_end,forecast_start,forecast_end):         
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]

                model.fit(X_train, y_train)
                #write training predicted
                y_train_pred=model.predict(X_train)
                df = pd.DataFrame(y_train_pred[:,:X.shape[1]], columns = asvid_list[n],index=range(train_start_i+time_steps,train_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.train_predicted.asv.csv') 

                num_iterations = y_test.shape[0]
                
                ## model evaluation for training
                y_train_pred = model.predict(X_train)
                mse = mean_squared_error(y_train, y_train_pred)
                mae = mean_absolute_error(y_train, y_train_pred)
                rmse = np.sqrt(mse)
                training_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
                
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    # Predict the next time point
                    next_prediction = model.predict(X_test)
                    next_time=int(X_test.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_test = np.concatenate((X_test[:,next_time:,:], next_prediction.reshape(1,1,-1)),axis=1)

                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                                
                ## model evaluation for test
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                test_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
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
                    X_last = np.concatenate((X_last[:,next_time:,:], next_prediction.reshape(1,1,-1)),axis=1)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                if forecast_start_i>X.shape[0]:
                    df = pd.DataFrame(y_pred[(forecast_start_i-X.shape[0]-1):(forecast_end_i-X.shape[0]),:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                else:
                    df = pd.DataFrame(y_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.forecasted.asv.csv')                
                n+=1
        elif model_name == "ARIMA":
            n = 0
            for arima_models, data, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(
                    model_data['model'], model_data['data'], train_start, train_end, test_start, test_end, forecast_start, forecast_end):
                X_train, X_test, y_train, y_test,X = data[0], data[1], data[2], data[3],data[4]
                y_train_pred = []
                y_test_pred = []

                # Predict training data
                for i, arima_model in enumerate(arima_models):
                    train_pred = arima_model.predict(start=0, end=len(X_train) - 1)
                    y_train_pred.append(train_pred)
                
                # Predict test data
                for i, arima_model in enumerate(arima_models):
                    test_pred = arima_model.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1)
                    y_test_pred.append(test_pred)
                
                y_train_pred = np.array(y_train_pred).T
                y_test_pred = np.array(y_test_pred).T
               
                # Save training predictions
                df_train = pd.DataFrame(y_train_pred[time_steps:,:X.shape[1]], columns=asvid_list[n], index=range(train_start_i + time_steps, train_end_i + 1))
                df_train.to_csv(f'Dataset.{n}.ARIMA.predictors.{predictor}.train_predicted.asv.csv')

                # Save test predictions
                df_test = pd.DataFrame(y_test_pred[:,:X.shape[1]], columns=asvid_list[n], index=range(test_start_i, test_end_i + 1))
                df_test.to_csv(f'Dataset.{n}.ARIMA.predictors.{predictor}.test_predicted.asv.csv')

                # Training error calculation
                mse_train = mean_squared_error(y_train, y_train_pred)
                mae_train = mean_absolute_error(y_train, y_train_pred)
                rmse_train = np.sqrt(mse_train)
                training_errors[model_name].append({'Dataset': n, 'MSE': mse_train, 'MAE': mae_train, 'RMSE': rmse_train})

                # Test error calculation
                mse_test = mean_squared_error(y_test, y_test_pred)
                mae_test = mean_absolute_error(y_test, y_test_pred)
                rmse_test = np.sqrt(mse_test)
                test_errors[model_name].append({'Dataset': n, 'MSE': mse_test, 'MAE': mae_test, 'RMSE': rmse_test})

                # Forecasting
                y_forecast_pred = []
                for i, arima_model in enumerate(arima_models):
                    forecast = arima_model.predict(start=forecast_start_i-1, end=forecast_end_i-1)
                    y_forecast_pred.append(forecast)

                y_forecast_pred = np.array(y_forecast_pred).T

                # Save forecasted predictions
                df_forecast = pd.DataFrame(y_forecast_pred[:, :X.shape[1]],
                                               columns=asvid_list[n], index=range(forecast_start_i, forecast_end_i + 1))

                df_forecast.to_csv(f'Dataset.{n}.ARIMA.predictors.{predictor}.forecasted.asv.csv')

                n += 1

        elif model_name == "VAR":
            n = 0
            for var_fitted, data, train_start_i, train_end_i, test_start_i, test_end_i,forecast_start_i, forecast_end_i in zip(
                    model_data['model'], model_data['data'], train_start, train_end, test_start, test_end,forecast_start, forecast_end):
                X_train, X_test, X = data[0], data[1],data[2]

                # Predict the training set using the model
                y_train_pred = var_fitted.fittedvalues
                
                # Predict the test set by forecasting
                y_test_pred = var_fitted.forecast(X_train, steps=len(X_test))

                # Save training predictions
                df_train = pd.DataFrame(y_train_pred[:,:X.shape[1]], columns=asvid_list[n], index=range(train_start_i + time_steps, train_end_i + 1))
                df_train.to_csv(f'Dataset.{n}.VAR.predictors.{predictor}.train_predicted.asv.csv')

                # Save test predictions
                df_test = pd.DataFrame(y_test_pred[:,:X.shape[1]], columns=asvid_list[n], index=range(test_start_i, test_end_i + 1))
                df_test.to_csv(f'Dataset.{n}.VAR.predictors.{predictor}.test_predicted.asv.csv')

                # Training error calculation
                mse_train = mean_squared_error(X_train[time_steps:,:X.shape[1]], y_train_pred[:,:X.shape[1]])
                mae_train = mean_absolute_error(X_train[time_steps:,:X.shape[1]], y_train_pred[:,:X.shape[1]])
                rmse_train = np.sqrt(mse_train)
                training_errors[model_name].append({'Dataset': n, 'MSE': mse_train, 'MAE': mae_train, 'RMSE': rmse_train})

                # Test error calculation
                mse_test = mean_squared_error(X_test[:,:X.shape[1]], y_test_pred[:,:X.shape[1]])
                mae_test = mean_absolute_error(X_test[:,:X.shape[1]], y_test_pred[:,:X.shape[1]])
                rmse_test = np.sqrt(mse_test)
                test_errors[model_name].append({'Dataset': n, 'MSE': mse_test, 'MAE': mae_test, 'RMSE': rmse_test})

                # Forecasting
                if forecast_start_i > X_train.shape[0]:
                    num_iterations = forecast_end_i - X_train.shape[0]
                    y_forecast_pred = var_fitted.forecast(X_train,steps=num_iterations)
                else:
                    num_iterations = forecast_end_i - forecast_start_i + 1
                    y_forecast_pred = var_fitted.forecast(X_train[:forecast_start_i,:],steps=num_iterations)

               
                # Save forecasted predictions
                if forecast_start_i>X_train.shape[0]:
                    df = pd.DataFrame(y_forecast_pred[(forecast_start_i-X_train.shape[0]-1):(forecast_end_i-X_train.shape[0]),:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                else:
                    df = pd.DataFrame(y_forecast_pred[:,:X.shape[1]], columns = asvid_list[n], index=range(forecast_start_i,forecast_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.forecasted.asv.csv')    
                n += 1

        ## Pytorch DNN, RNN, LSTM model       
        elif model_name in ["pytorch_DNN","pytorch_RNN","pytorch_LSTM"]:
            n=0
          # Assuming data[0] is the input tensor and data[2] is the target tensor
            batch_size = 100
            if model_name in ["pytorch_DNN"]:
              epochs = config['pytorch_DNN']['epoch']
            elif model_name in ["pytorch_RNN"]:
              epochs = config['pytorch_RNN']['epoch']
            elif model_name in ["pytorch_LSTM"]:
              epochs = config['pytorch_LSTM']['epoch']
                          
            for model_set,data,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['model'],
                                                                                          model_data['data'],test_start,test_end,forecast_start,forecast_end):
                model, optimizer, loss_fn=model_set
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]
                #scaler = StandardScaler()
                #X_train = scaler.fit_transform(X_train)
                #X_test = scaler.transform(X_test)
                #X_last = scaler.transform(X_last)
                dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                pytorch_module.train_pytorch_model(model, optimizer, loss_fn, dataloader, epochs)
        
                
                # Switch to evaluation mode for prediction
                model.eval()
                
                ## evaluation of traininig
                y_train_pred = model(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
                mse = mean_squared_error(y_train, y_train_pred)
                mae = mean_absolute_error(y_train, y_train_pred)
                rmse = np.sqrt(mse)
                training_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
                
                # Iterative predictions for test set
                y_pred =  pytorch_module.evaluate_pytorch_model(model,model_name, X_test, y_test, time_steps)
                                
                ## model evaluation for test
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                test_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
                df = pd.DataFrame(y_pred[:, :X.shape[1]], columns=asvid_list, index=range(test_start_i, test_end_i + 1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.predicted.asv.csv')
                
                # Iterative predictions for forecast set
                X_last_tensor = torch.tensor(X_last, dtype=torch.float32)
                y_pred = []
                
                if forecast_start_i > X.shape[0]:
                    num_iterations = forecast_end_i - X.shape[0]
                else:
                    num_iterations = forecast_end_i - forecast_start_i + 1
                
                with torch.no_grad():
                    for i in range(num_iterations):
                        X_last_tensor = torch.tensor(X_last, dtype=torch.float32)
                        next_prediction = model(X_last_tensor).numpy()
                        next_time = int(X_last_tensor.shape[1] / time_steps)
                        if model_name in ['pytorch_DNN']:
                          X_last = np.concatenate((X_last[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                        elif model_name in ['pytorch_RNN','pytorch_LSTM']:
                          X_last = np.concatenate((X_last[:,next_time:,:], next_prediction.reshape(1,1,-1)),axis=1)
                        y_pred.append(next_prediction.reshape(-1))
                
                y_pred = np.array(y_pred)
                if forecast_start_i > X.shape[0]:
                    df = pd.DataFrame(y_pred[(forecast_start_i - X.shape[0] - 1):(forecast_end_i - X.shape[0]), :X.shape[1]], columns=asvid_list, index=range(forecast_start_i, forecast_end_i + 1))
                else:
                    df = pd.DataFrame(y_pred[:, :X.shape[1]], columns=asvid_list, index=range(forecast_start_i, forecast_end_i + 1))
                
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.forecasted.asv.csv')
                n+=1 
        ## For PCR and PLS model
        elif model_name in ["PCR","PLS"]:
            n=0
            for model,data,train_start_i,train_end_i,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['model'],model_data['data'],train_start,train_end,test_start,test_end,forecast_start,forecast_end):         
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]
                model.fit(X_train, y_train)
                #write training predicted
                y_train_pred=model.predict(X_train)
                df = pd.DataFrame(y_train_pred[:,:X.shape[1]], columns = asvid_list[n],index=range(train_start_i+time_steps,train_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.train_predicted.asv.csv') 

                ## Model evaluation of training processes
                y_train_pred = model.predict(X_train)
                mse = mean_squared_error(y_train, y_train_pred)
                mae = mean_absolute_error(y_train, y_train_pred)
                rmse = np.sqrt(mse)
                training_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
                
                num_iterations = y_test.shape[0]
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    #print("X_test[:,-10:] for "+str(i)+"th iteration")
                    #print(X_test[:,-10:])
                    # Predict the next time point
                    next_prediction = model.predict(X_test)
                    # print(next_prediction)
                    next_time=int(X_test.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_test = np.concatenate((X_test[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                    # print(X_test)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                                
                ## model evaluation for test
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                test_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                    
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
                training_errors_df = pd.DataFrame(training_errors)        
        ## For other sklearn models
        else:
            n=0
            model=model_data['model']
            for data,train_start_i,train_end_i,test_start_i,test_end_i,forecast_start_i,forecast_end_i in zip(model_data['data'],train_start,train_end,test_start,test_end,forecast_start,forecast_end):         
                X_train,X_test,y_train,y_test,X_last,X=data[0],data[1],data[2],data[3],data[4],data[5]
                model.fit(X_train, y_train)
                #write training predicted
                y_train_pred=model.predict(X_train)
                df = pd.DataFrame(y_train_pred[:,:X.shape[1]], columns = asvid_list[n],index=range(train_start_i+time_steps,train_end_i+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.train_predicted.asv.csv') 

                ## Model evaluation of training processes
                y_train_pred = model.predict(X_train)
                mse = mean_squared_error(y_train, y_train_pred)
                mae = mean_absolute_error(y_train, y_train_pred)
                rmse = np.sqrt(mse)
                training_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                
                
                num_iterations = y_test.shape[0]
                # Make iterative predictions
                y_pred=[]
                for i in range(num_iterations):
                    #print("X_test[:,-10:] for "+str(i)+"th iteration")
                    #print(X_test[:,-10:])
                    # Predict the next time point
                    next_prediction = model.predict(X_test)
                    # print(next_prediction)
                    next_time=int(X_test.shape[1]/time_steps)
                    # Update feature matrix X by shifting down and adding the predicted value
                    X_test = np.concatenate((X_test[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
                    # print(X_test)
                    # Update target array y by shifting down
                    y_pred.append(next_prediction.reshape(-1))
                y_pred=np.array(y_pred)
                                
                ## model evaluation for test
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                test_errors[model_name].append({'Dataset': n, 'MSE': mse, 'MAE': mae, 'RMSE': rmse})
                    
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
                
    train_error = []

    # Loop through the dictionary and append model names and metrics to the list
    for model_name, metrics_list in training_errors.items():
        for metrics in metrics_list:
            # Create a copy of the dictionary to avoid modifying the original
            entry = metrics.copy()
            # Add the model name to the entry
            entry['Model'] = model_name
            # Append the entry to the data list
            train_error.append(entry)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(train_error)

    # Save the DataFrame to a CSV file
    df.to_csv('training_errors.csv', index=False)

    test_error = []

    # Loop through the dictionary and append model names and metrics to the list
    for model_name, metrics_list in test_errors.items():
        for metrics in metrics_list:
            # Create a copy of the dictionary to avoid modifying the original
            entry = metrics.copy()
            # Add the model name to the entry
            entry['Model'] = model_name
            # Append the entry to the data list
            test_error.append(entry)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(test_error)

    # Save the DataFrame to a CSV file
    df.to_csv('test_errors.csv', index=False)
    print("training errors and test errors saved into output")


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


    
    

