import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Create a Model Class that inherits nn.Module
class Model(nn.Module):

    # Input layer (23 parameters) --> hidden layers --> Output layer
    def __init__(self, in_features=23, h1=10, h2=10, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    # Method to move forward data
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# Pick a manual seed for randomization
torch.manual_seed(41)
model = Model()

# Dataframe
df = pd.read_csv('MushroomsModel\mushrooms.csv')

# X are the features so we drop all the outcome colums
X = df.drop('class', axis=1)
y = df['class']

# Convert X and y to numpy arrays
X = X.values
y = y.values

# Train Test Split (Train size 80% of dataset, Test size 20% of dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X to char tensors
X_train = torch.CharTensor(X_train)
X_test = torch.CharTensor(X_test)

# Convert X to char tensors
y_train = torch.CharTensor(y_train)
y_test = torch.CharTensor(y_test)

