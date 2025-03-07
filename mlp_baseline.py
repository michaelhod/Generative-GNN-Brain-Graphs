import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from evaluation import evaluate_matrices
from MatrixVectorizer import MatrixVectorizer


class NaiveMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(NaiveMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train_model(model, X_train, y_train, X_val, y_val):
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 64

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()  # or another loss function appropriate for your task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val_tensor)
        val_loss = criterion(y_pred, y_val_tensor)
    
    # Convert prediction vectors back to matrices
    # The shape of the y_pred_matrix, y_val_matrix should be (n_samples, adj_size, adj_size)
    print("Converting predictions to matrices...")
    y_pred_matrix = np.array([MatrixVectorizer.anti_vectorize(y_pred[i], matrix_size=HR_Adj_shape, include_diagonal=False) for i in range(y_pred.shape[0])])
    y_val_matrix = np.array([MatrixVectorizer.anti_vectorize(y_val_tensor[i], matrix_size=HR_Adj_shape, include_diagonal=False) for i in range(y_val_tensor.shape[0])])

    y_val_matrix[y_val_matrix < 0] = 0

    print("Evaluating matrices...")
    metrics = evaluate_matrices(y_pred_matrix, y_val_matrix, all_metrics=False)
    metrics['val_loss'] = val_loss.item()

    return model, metrics

def train_all_data(model, X_train, y_train):
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 64

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    return model


if __name__ == "__main__":
    LR_TRAIN_DATA_FILE_NAME = "lr_train.csv"
    LR_TEST_DATA_FILE_NAME = "lr_test.csv"
    HR_TRAIN_DATA_FILE_NAME = "hr_train.csv"

    LR_TRAIN_DATA_PATH = os.path.join("data", LR_TRAIN_DATA_FILE_NAME)
    LR_TEST_DATA_PATH = os.path.join("data", LR_TEST_DATA_FILE_NAME)
    HR_TRAIN_DATA_PATH = os.path.join("data", HR_TRAIN_DATA_FILE_NAME)

    df_lr_train = pd.read_csv(LR_TRAIN_DATA_PATH)
    df_lr_test = pd.read_csv(LR_TEST_DATA_PATH)
    df_hr_train = pd.read_csv(HR_TRAIN_DATA_PATH)

    LR_Adj_shape = 160
    HR_Adj_shape = 268

    k_folds = 3
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    X = df_lr_train
    y = df_hr_train

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{k_folds}")
        
        # Split data for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Validation samples: {X_val.shape[0]}")
        
        model = NaiveMLP(
            input_dim=12720,
            hidden_dims=[4096, 2048, 1024, 512],
            output_dim=35778,
            dropout_rate=0.3
        )

        model, metrics = train_model(model, X_train, y_train, X_val, y_val)
        fold_metrics.append(metrics)

        print(f"  Validation Loss: {metrics['val_loss']:.4f}")
    print(fold_metrics)

    # # Train on all data
    # model = NaiveMLP(
    #     input_dim=12720,
    #     hidden_dims=[4096, 2048, 1024, 512],
    #     output_dim=35778,
    #     dropout_rate=0.3
    # )
    # model = train_all_data(model, X, y)
    # # Save model
    # torch.save(model.state_dict(), "model.pth")
    # print("Model saved")