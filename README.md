# Predict microbial composition of time-series
Runing the __src/main.py__ can fit a few machine learning models for the asv abundance and environmental factors using training data, and predict the asv abundance using test data.

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

The output file name, for example, Dataset.0.DNN.predictors.env.predicted.asv.csv.  Dataset.0(indicate the first dataset in __map.tsv__ file).DNN.(the model use DNN model)predictors.env(the predictor is environmental variables).predicted.asv.csv


### Change the setting of model and datasets
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


test_end=last     
## last or a number smaller than the minium size of all datasets

train_size=all    
## all or int <= test_end-test_size

test_size=8

## int, < sample size

env_num=all      
## all or int, the first n environmental variables as predictors.

time_steps=1        
## int,steps of previous time points as predictor

for_periods=1
# steps of latter time points as y, it matters for RNN, LSTM

```


