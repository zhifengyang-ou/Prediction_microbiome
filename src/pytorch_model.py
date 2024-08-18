# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:09:54 2024

@author: 10051
"""

import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_channels, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 24, 100)  # adjust the dimensions according to your data
        self.fc2 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 24)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

from torch.utils.data import DataLoader, TensorDataset

# Define a function to train PyTorch models
def train_pytorch_model(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

dnn=FeedforwardNN(input_dim=10, hidden_dim=10, output_dim=1)
print(dnn)

cnn=CNN(num_channels=10, output_dim=1)
print(cnn)

lstm=LSTM(input_dim=10, hidden_dim=10, num_layers=10, output_dim=1)
print(lstm)