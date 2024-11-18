
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Name: main.py
Description: This script fits a few machine learning models using data of ASV abundance and environmental variables
to predict time-series ASV abundance. Updated to include GRU and Transformer models.
Author: Zhifeng Yang (original), Updated by ChatGPT
"""

# Import necessary modules/packages
import configparser
import os
import pandas as pd
import numpy as np

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Sklearn Models
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Keras models
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM, MultiHeadAttention, LayerNormalization, Input
from keras.models import Model
from scikeras.wrappers import KerasRegressor
from keras.optimizers import Adam

# ARIMA model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

## Custom functions and helper code omitted for brevity ##

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

def create_transformer_regressor(input_dimension, sequence_length, output_dimension, n_heads, n_units, n_layers, dropout, learning_rate):
    """ Create Transformer model """
    inputs = Input(shape=(sequence_length, input_dimension))
    x = inputs
    for _ in range(n_layers):
        attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=n_units)(x, x)
        x = LayerNormalization()(x + attn_output)
        ffn_output = Dense(n_units, activation='relu')(x)
        x = LayerNormalization()(x + ffn_output)
    outputs = Dense(output_dimension, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

def build_models_and_split_data(env_list, asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end, config):
    """ Updated to include GRU and Transformer models """
    data_model = {}
    model_names = config['models']

    if 'GRU' in model_names:
        data_model['GRU'] = {'model': [], 'data': []}
        for asv, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(
                asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end):
            X_train, y_train, X_test, y_test = split_sample_ts(asv, train_start_i, train_end_i, test_start_i, test_end_i, time_steps, for_periods, reshape=False)
            model = KerasRegressor(
                model=create_gru_regressor(input_dimension=X_train.shape[2], sequence_length=X_train.shape[1], output_dimension=y_train.shape[1],
                                           n_layer=config['GRU']['n_layer'], n_unit=config['GRU']['n_unit'], dropout=config['GRU']['dropout'], learning_rate=config['GRU']['learning_rate']),
                epochs=config['GRU']['epoch'], batch_size=32, verbose=0
            )
            data_model['GRU']['model'].append(model)
            data_model['GRU']['data'].append([X_train, X_test, y_train, y_test])

    if 'Transformer' in model_names:
        data_model['Transformer'] = {'model': [], 'data': []}
        for asv, train_start_i, train_end_i, test_start_i, test_end_i, forecast_start_i, forecast_end_i in zip(
                asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end):
            X_train, y_train, X_test, y_test = split_sample_ts(asv, train_start_i, train_end_i, test_start_i, test_end_i, time_steps, for_periods, reshape=False)
            model = KerasRegressor(
                model=create_transformer_regressor(input_dimension=X_train.shape[2], sequence_length=X_train.shape[1], output_dimension=y_train.shape[1],
                                                   n_heads=config['Transformer']['n_heads'], n_units=config['Transformer']['n_units'],
                                                   n_layers=config['Transformer']['n_layers'], dropout=config['Transformer']['dropout'], learning_rate=config['Transformer']['learning_rate']),
                epochs=config['Transformer']['epoch'], batch_size=32, verbose=0
            )
            data_model['Transformer']['model'].append(model)
            data_model['Transformer']['data'].append([X_train, X_test, y_train, y_test])

    # Additional models handled here...

    return data_model

if __name__ == "__main__":
    config = read_config()
    env_files, asv_files, train_start, train_end, test_start, test_end, forecast_start, forecast_end = read_map_file(config)
    env_list, asv_list, asvid_list = read_data(env_files, asv_files, config)
    data_model = build_models_and_split_data(env_list, asv_list, train_start, train_end, test_start, test_end, forecast_start, forecast_end, config)
    models_fit_predict(data_model, asvid_list, test_start, test_end, forecast_start, forecast_end, config)
