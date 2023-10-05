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
import glob
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
from sklearn.feature_selection import SelectFromModel
## keras models
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

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
    return {
        'dir_env': config['Paths']['dir_env'],
        'dir_asv': config['Paths']['dir_asv'],
        'dir_output':config['Paths']['dir_output'],
        'map_file':config['Paths']['map_file'],
        'predictor':config['model_setting']['predictor'],
        'models':models,
        'test_end':config['model_setting']['test_end'],
        'train_size':config['model_setting']['train_size'],
        'test_size':config['model_setting']['test_size'],
        'env_num':config['model_setting']['env_num'],
        'time_steps':config['model_setting']['time_steps'],
        'for_periods':config['model_setting']['for_periods']
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

    with open(file_path, 'r', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            env_files.append(row[0])
            asv_files.append(row[1])

    return env_files[1:],asv_files[1:]

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
def split_sample_ts(all_data,test_end,train_size,test_size,time_steps,for_periods,reshape=True):
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
    ts_train = all_data[(test_end-test_size-train_size):(test_end-test_size),:]
    ts_test  = all_data[(test_end-test_size):test_end,:]
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # create training data of {train_size-time_steps-for_periods+1} samples and {time_steps} time steps for X_train
    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps,ts_train_len-for_periods+1): 
        X_train.append(ts_train[i-time_steps:i,:])
        y_train.append(ts_train[i+for_periods-1,:])
    if reshape:
        X_train, y_train = np.array(X_train).reshape(train_size-time_steps-for_periods+1,-1), np.array(y_train).reshape(train_size-time_steps-for_periods+1,-1)
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
        X_test = np.array(X_test).reshape(test_size,-1)
        y_test=np.array(y_test).reshape(test_size,-1)
    else:
        X_test = np.array(X_test)
        y_test=np.array(y_test) 

    return X_train, y_train , X_test, y_test




## build models and associated data
## now only include models and data, may include optimizer later--zhifeng

def build_models_and_split_data(env_list,asv_list,config):
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
    models=[]
    data_model={}   ## need define a class for the structure (includes data, model, optimizer, etc.) later--zhifeng
    data_list=[]
    model_names=config['models']
    predictor=config['predictor']
    test_end=config['test_end']     
    train_size=config['train_size']   
    test_size=int(config['test_size'])  
    env_num=config['env_num']  
    time_steps=int(config['time_steps'])  
    for_periods=int(config['for_periods'])  
    
    ## split data based on the selection of predictors
    if predictor == "env":   ## use current environmental variables to predict asv abundance
        for env, asv in zip(env_list,asv_list):
            if test_end=="last":
                test_end=asv.shape[0]
            elif test_end>asv.shape[0]:
                test_end=asv.shape[0]
                print("The test end is larger than the sample size, so use the last time point as the end of test set")
            if env_num=="all":
                env_num=env.shape[1]
            elif env_num>env.shape[1]:
                env_num=env.shape[1]  
                print("The number of enviromental factors is larger than all the number, so use all the environmental factors as X")
            if train_size=="all":
                train_size=test_end-test_size
            elif train_size > test_end-test_size:
                train_size=test_end-test_size
                print("The train size is too large, so use the test_end-test_size as the train_size")
                
            X_train,X_test=split_samples(env[:test_end,:env_num],train_size=train_size,test_size=test_size)
            y_train,y_test=split_samples(asv[:test_end,:],train_size=train_size,test_size=test_size)
            data_list.append([X_train,X_test,y_train,y_test])
    if predictor == "env+asv":  ## use previous asv abundance and current environmental variables to predict asv abundance
        for env, asv in zip(env_list,asv_list):
            if test_end=="last":
                test_end=asv.shape[0]
            elif test_end>asv.shape[0]:
                test_end=asv.shape[0]
                print("The test end is larger than the sample size, so use the last time point as the end of test set")
            if env_num=="all":
                env_num=env.shape[1]
            elif env_num>env.shape[1]:
                env_num=env.shape[1]  
                print("The number of enviromental factors is larger than all the number, so use all the environmental factors as X")
            if train_size=="all":
                train_size=test_end-test_size
            elif train_size > test_end-test_size:
                train_size=test_end-test_size
                print("The train size is too large, so use the test_end-test_size as the train_size")           
            ## split asv abundance as X and Y
            X_train, y_train , X_test, y_test=split_sample_ts(asv,test_end,train_size,test_size,time_steps,for_periods)
            ## split environmental variables as X
            X_train2,X_test2=split_samples(env[:test_end,:env_num],train_size=train_size-time_steps-for_periods+1,test_size=test_size)
            y_train2,y_test2=split_samples(asv[:test_end,:],train_size=train_size-time_steps-for_periods+1,test_size=test_size)

            X_train=np.concatenate((X_train,X_train2),axis=1)
            X_test=np.concatenate((X_test,X_test2),axis=1)
            data_list.append([X_train,X_test,y_train,y_test])
            
    ## define models and stored it with its datasets
    if "Dummy" in model_names:
        dummy=make_pipeline(DummyRegressor(strategy="mean"))
        data_model['Dummy']={'model':dummy,'data':data_list}

    if 'LinearRegression' in model_names:    
        lr=make_pipeline(StandardScaler(),LinearRegression())
        data_model['LinearRegression']={'model':lr,'data':data_list}
    if 'Ridge' in model_names:
        ridge=make_pipeline(StandardScaler(),Ridge())
        data_model['Ridge']={'model':ridge,'data':data_list}    
    if 'PLS' in model_names:
        pls=make_pipeline(StandardScaler(),PLSRegression(5))
        data_model['PLS']={'model':pls,'data':data_list} 
    if 'PCR' in model_names:
        pcr = make_pipeline(StandardScaler(), PCA(n_components=5),LinearRegression())
        data_model['PCR']={'model':pcr,'data':data_list}
    if 'RandomForest' in model_names:
        rf=make_pipeline(StandardScaler(),RandomForestRegressor(random_state=0))
        data_model['RandomForest']={'model':rf,'data':data_list}
    if 'MLP' in model_names:
        mlp=make_pipeline(StandardScaler(),MLPRegressor(random_state=1,  learning_rate_init=0.01, max_iter=10000))
        data_model['MLP']={'model':mlp,'data':data_list}     
    ## For different datasets, we need build different keras models as the input dimension is dependent on the dimension of X
    if 'DNN' in model_names:
        data_model['DNN']={'model':[],'data':data_list} 
        for data in data_model['DNN']['data']:
            input_dimension = data[0].shape[1] # Replace with the appropriate input dimension
            output_dimension=data[2].shape[1]
            def create_dnn_regressor(input_dimension=input_dimension,output_dimension=output_dimension):
                model = Sequential()
                model.add(Dense(units=64, activation='relu', input_dim=input_dimension))
                model.add(Dense(units=32, activation='relu'))  # Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
                return model
            model = KerasRegressor(build_fn=create_dnn_regressor, epochs=500, batch_size=32)
            data_model['DNN']['model'].append(model)
     
    ## For RNN and LSTM models, only previous time points of asv abundance can be used as predictor
    if 'RNN' in model_names:
        data_model['RNN']={'model':[],'data':[]} 
        for asv in asv_list:
            if test_end=="last":
                test_end=asv.shape[0]
            elif test_end>asv.shape[0]:
                test_end=asv.shape[0]
                print("The test end is larger than the sample size, so use the last time point as the end of test set")
            if train_size=="all":
                train_size=test_end-test_size
            elif train_size > test_end-test_size:
                train_size=test_end-test_size
                print("The train size+test_size exeeds sample size, so use the test_end-test_size as the train_size")
            X_train, y_train , X_test, y_test=split_sample_ts(asv,test_end,train_size,test_size,time_steps,for_periods,reshape=False)
            print(X_train.shape)
            data_model['RNN']['data'].append([X_train,X_test,y_train,y_test])
            # Specify the input dimension (number of features) and sequence length
            input_dimension = X_train.shape[2]# Replace with the appropriate input dimension
            sequence_length = X_train.shape[1]# Replace with the appropriate sequence length
            output_dimension=y_train.shape[1]
            def create_rnn_regressor(input_dimension=input_dimension,sequence_length=sequence_length,output_dimension=output_dimension):
                from keras.layers import SimpleRNN
                model = Sequential()
                model.add(SimpleRNN(units=64, activation='relu', input_shape=(sequence_length, input_dimension)))
                model.add(Dense(units=32, activation='relu'))  # Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
                return model


            # Create a KerasRegressor instance using your RNN regressor model function
            model = KerasRegressor(build_fn=create_rnn_regressor, 
                                   epochs=200, batch_size=32)
            data_model['RNN']['model'].append(model)
    if 'LSTM' in model_names:
        data_model['LSTM']={'model':[],'data':[]} 
        for asv in asv_list:
            if test_end=="last":
                test_end=asv.shape[0]
            elif test_end>asv.shape[0]:
                test_end=asv.shape[0]
                print("The test end is larger than the sample size, so use the last time point as the end of test set")
            if train_size=="all":
                train_size=test_end-test_size
            elif train_size > test_end-test_size:
                train_size=test_end-test_size
                print("The train size is too large, so use the test_end-test_size as the train_size")
            X_train, y_train , X_test, y_test=split_sample_ts(asv,test_end,train_size,test_size,time_steps,for_periods,reshape=False)
            data_model['LSTM']['data'].append([X_train,X_test,y_train,y_test])
            # Specify the input dimension (number of features) and sequence length
            input_dimension = X_train.shape[2]# Replace with the appropriate input dimension
            sequence_length = X_train.shape[1]# Replace with the appropriate sequence length
            output_dimension=y_train.shape[1]
            # Create a KerasRegressor instance using your RNN regressor model function
            def create_lstm_regressor(input_dimension=input_dimension,sequence_length=sequence_length,
                                      output_dimension=output_dimension):
                from keras.layers import LSTM
                model = Sequential()
                model.add(LSTM(units=64, activation='relu', input_shape=(sequence_length, input_dimension)))
                model.add(Dense(units=32, activation='relu'))  # Additional hidden layer
                model.add(Dense(units=output_dimension, activation='linear'))  # Output layer for regression (linear activation)
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
                return model

            model = KerasRegressor(build_fn=create_lstm_regressor, epochs=200, batch_size=32)
            data_model['LSTM']['model'].append(model)
    return data_model



## model fit and predict
def models_fit_predict(data_model,asvid_list,config):
    """
    usage:
        Fit a list of scikit-learn and keras models with training data and predict asv abundance for test data
    -------
    return:
    generate predicted asv table in output directory

    """
    dir_output=config['dir_output']
    predictor=config['predictor']
    test_end=config['test_end']
    test_size=int(config['test_size'])  
    time_steps=int(config['time_steps'])  
    for_periods=int(config['for_periods']) 
    
    os.chdir(dir_output)  # set dir location
    for model_name, model_data in data_model.items():

        if model_name in ["DNN","RNN","LSTM"]:
            n=0
            for model,data in zip(model_data['model'],model_data['data']):         
                X_train,X_test,y_train,y_test=data[0],data[1],data[2],data[3]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if test_end=="last":
                    if predictor=="env":
                        test_end=data[0].shape[0]+data[1].shape[0]
                    else:
                        test_end=data[0].shape[0]+data[1].shape[0]+time_steps+for_periods-1
                df = pd.DataFrame(y_pred, columns = asvid_list[n], index=range(test_end-test_size+1,test_end+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.predicted.asv.csv')
                n+=1
        else:
            n=0
            model=model_data['model']
            for data in model_data['data']:         
                X_train,X_test,y_train,y_test=data[0],data[1],data[2],data[3]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if test_end=="last":
                    if predictor=="env":
                        test_end=data[0].shape[0]+data[1].shape[0]
                    else:
                        test_end=data[0].shape[0]+data[1].shape[0]+time_steps+for_periods-1
                df = pd.DataFrame(y_pred, columns = asvid_list[n], index=range(test_end-test_size+1,test_end+1))
                df.to_csv('Dataset.'+str(n)+'.'+model_name+'.predictors.'+str(predictor)+'.predicted.asv.csv')
                n+=1 




if __name__ == "__main__":
    config = read_config()
    
    print("Model configuration:")
    print(f"Models: {config['models']}")
    print(f"Predictors: {config['predictor']}")
    print(f"test_end: {config['test_end']}")
    print(f"train_size: {config['train_size']}")    
    print(f"test_size: {config['test_size']}")
    print(f"env_num: {config['env_num']}")
    print(f"time_steps: {config['time_steps']}")
    print(f"for_periods: {config['for_periods']}")
    
    print("---------------------")    
    print("reading map file...")
    env_files,asv_files=read_map_file(config)
    print("reading data...")
    env_list, asv_list,asvid_list=read_data(env_files,asv_files,config)
    print("building models...")
    data_model=build_models_and_split_data(env_list,asv_list,config)
    print("model fitting and predicting...")
    models_fit_predict(data_model,asvid_list,config)
    
    print("---------------------")
    print(f"predicted asv tables are stored in {config['dir_output']}")


    
    

