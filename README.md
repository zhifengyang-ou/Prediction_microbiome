# Predict microbial composition of time-series
Runing the __src/main.py__ can fit a few machine learning models for the asv abundance and environmental factors using training data, and predict the asv abundance using test data.

## Update 2/13/2024
Since requirement.yaml contains system-specific information, so I added an requirement.yaml without specific system requirement for installation.

## Update 1/12/24
Issue:

Encountered sensitivity of MLPRegressor to the scale of input features during iterative predictions. The problem manifested when predicted values became the next input for prediction, leading to errors, especially when the predicted values exceeded a certain threshold.

Solution:        
Scaling Strategy Change:
Initially used StandardScaler(), but it was sensitive to variations between training and new data.              
Switched to QuantileTransformer() for scaling, ensuring transformation to a fixed range [0, 1].        
QuantileTransformer handles outliers by assigning boundary values, preventing predicted values from becoming excessively large.        
        
I also made a Correction for Multiple Datasets:
Addressed an issue where, when using multiple datasets in a map file, the environmental variable number for X was fixed to the first dataset's values.
Updated the implementation to allow each dataset to utilize its respective environmental variable numbers independently.

## Update 12/23/23
Now, you can customize the settings of hyperparameters for different models by adjusting the configuration file. While not all model hyperparameters are configurable through this file, I have included some crucial ones based on my knowledge. To explore the range and data type of hyperparameters, please refer to the additional /src/config_illustration.ini file. I separated the illustration from the original .ini file to address certain encoding issues in the introduction. For a more comprehensive overview of model settings and a wider range of possible hyperparameters to tweak, please see the reference link provided in the /src/config_illustration.ini file.

__ps:__ the n_layer parameter in DNN, RNN and LSTM configuration is not a specific hyperparameters for the model functions, but controls the number of hidden layers in the neural network.

## Update 12/11/23
Now the model only use previous asv or previous asv+ previous env as predictor to predict next 1 time point of asv abundance in model fitting.
Now the training period, test period,and forcast period could be set for each data set in the map file.But be careful that the test_end is not larger than total data size and forecast_start is not smaller than total data size.
```
ENV	ASV	train_start	train_end	test_start	test_end	forecast_start	forecast_end
Air01A1.Env.fillna.csv	Air01A1.ASV.Top100.csv	1	30	31	40	50	70
Air01A1.Env.fillna.csv	Air01A1.ASV.Top1000.csv	5	30	31	40	50	70
```

When doing prediction, the prediction for test and forecast(the future points outside data range), the model make iterative prediction, that is, predicted next n time point will be included in predictor for next n+1 time point.
For forecast, I used the last several time points as predictor to iteratively predict future. Based on your selection of start point and end point of forecast, the predicted outcome will be extracted from the whole prediction (from {end data point +1} of data to end point of forecast).


## Installation

Clone the current project
```
git clone https://github.com/zhifengyang-ou/Prediction_microbiome.git
```
Open your command-line terminal and navigate to the directory where the __requirement.yaml__ file is located, creat a environment and install requirements
```
conda env create -f requirement.yaml -n myenv
```
Activate environment
```
conda activate myenv
```
## Usage
In the conda environment, navigate to __src/__ folder and run
```
cd src
python main.py
```
The output is in the __output/__ folder in default. The data is inside the __data/__ folder.

## Meaning of output file name
The output file name, for example, __Dataset.0.DNN.predictors.env.predicted.asv.csv__, indicates: __Dataset.0__(indicate the first dataset in __map.tsv__ file)__.DNN.__(the model use DNN model)__predictors.env__(the predictor is environmental variables).predicted.asv.csv
Now  __Dataset.0.DNN.predictors.env.forecasted.asv.csv__ is the predictions for future points.

## Change the setting of model and datasets
You can change the __src/map.tsv__ and __src/config.ini__ file to change the path of datasets, model setting and output dirctory.

The __src/map.tsv__ includes the paths of asv and env csv tables, you can add more rows of pairs of environmental and asv table files.
```
ENV	ASV
Air01A1.Env.fillna.csv	Air01A1.ASV.Top100.csv
Air01A1.Env.fillna.csv	Air01A1.ASV.Top1000.csv
```
The __src/config.ini__ includes the configuration information of __main.py__ script
```
## configuration file
### do not use quote here, see the annotation for the options.

[Paths]
dir_env = ../data
dir_asv= ../data
dir_output= ../output
map_file=map.tsv


[model_setting]
predictor= asv      
## or (env+asv); use (asv) for RNN and LSTM

models=RNN,LSTM
# option: Dummy,LinearRegression,Ridge,PLS,PCR,RandomForest,MLP,DNN, RNN,LSTM


env_num=all      
## all or int, the first n environmental variables as predictors.

time_steps=1        
## int,steps of previous time points as predictor

for_periods=1
# steps of latter time points as y, it matters for RNN, LSTM

```


