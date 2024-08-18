import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the PyTorch DNN model
class DNNRegressor(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_layer, n_unit, dropout):
        super(DNNRegressor, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dimension, n_unit))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(n_unit, n_unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(n_unit, output_dimension))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Function to create and return the model, optimizer, and loss function
def create_dnn_regressor(input_dimension, output_dimension, n_layer, n_unit, dropout,lr):
    model = DNNRegressor(input_dimension, output_dimension, n_layer, n_unit, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn


## Define the PyTorch RNN model

class PyTorchRNN(nn.Module):
    def __init__(self, input_dim, sequence_length, output_dim, n_layers, n_units, dropout):
        super(PyTorchRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=n_units, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(n_units * sequence_length, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sequence_length = sequence_length
        self.n_units = n_units

    def forward(self, x):
        h_0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device) # Initialize hidden state
        out, _ = self.rnn(x, h_0)
        out = out.contiguous().view(out.size(0), -1) # Flatten the output for the linear layer
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_pytorch_rnn_regressor(input_dim, sequence_length, output_dim, n_layers, n_units, dropout, learning_rate):
    model = PyTorchRNN(input_dim, sequence_length, output_dim, n_layers, n_units, dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn


## Define the PyTorch LSTM model
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, sequence_length, output_dim, n_layers, n_units, dropout):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_units, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(n_units * sequence_length, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sequence_length = sequence_length
        self.n_units = n_units

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device) # Initialize hidden state
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device) # Initialize cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = out.contiguous().view(out.size(0), -1) # Flatten the output for the linear layer
        out = self.dropout(out)
        out = self.fc(out)
        return out

def create_pytorch_lstm_regressor(input_dim, sequence_length, output_dim, n_layers, n_units, dropout, learning_rate):
    model = PyTorchLSTM(input_dim, sequence_length, output_dim, n_layers, n_units, dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn



## Define training function for pytorch model
def train_pytorch_model(model, optimizer, loss_fn, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.float(), targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss/len(dataloader)}')

def evaluate_pytorch_model(model,model_name, X_test, y_test, time_steps):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for i in range(y_test.shape[0]):
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            next_prediction = model(X_test_tensor).numpy()
            next_prediction = np.clip(next_prediction, -1e10, 1e10)
            next_time = int(X_test_tensor.shape[1] / time_steps)
            if model_name in ['pytorch_DNN']:
              X_test = np.concatenate((X_test[:,next_time:], next_prediction.reshape(1,-1)),axis=1)
            elif model_name in ['pytorch_RNN','pytorch_LSTM']:
              X_test = np.concatenate((X_test[:,next_time:,:], next_prediction.reshape(1,1,-1)),axis=1)
            y_pred.append(next_prediction.reshape(-1))
    y_pred = np.array(y_pred)
    return y_pred

