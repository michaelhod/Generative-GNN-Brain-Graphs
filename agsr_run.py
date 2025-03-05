"""Main function of Adversarial Graph Super-Resolution Network (AGSR-Net) framework for predicting high-resolution
    connectomes from low-resolution connectomes.
    ---------------------------------------------------------------------

    This file contains the implementation of the training and testing process of our AGSR-Net model.
        train(model, subjects_adj, subjects_ground_truth, args)
                Inputs:
                        model:        constructor of our AGSR-Net model:  model = AGSRNet(ks,args)
                                      ks:   array that stores reduction rates of nodes in Graph U-Net pooling layers
                                      args: parsed command line arguments
                        subjects_adj: (n × l x l) tensor stacking LR connectivity matrices of all training subjects
                                       n: the total number of training subjects
                                       l: the dimensions of the LR connectivity matrices
                        subjects_ground_truth: (n × h x h) tensor stacking LR connectivity matrices of all training subjects
                                                n: the total number of training subjects
                                                h: the dimensions of the LR connectivity matrices
                        args:          parsed command line arguments, to learn more about the arguments run:
                                       python demo.py --help
                Output:
                        for each epoch, prints out the mean training MSE error

        test(model, test_adj,test_ground_truth,args)
                Inputs:
                        test_adj:      (t × l x l) tensor stacking LR connectivity matrices of all testing subjects
                                        t: the total number of testing subjects
                                        l: the dimensions of the LR connectivity matrices
                        test_ground_truth:      (t × h x h) tensor stacking LR connectivity matrices of all testing subjects
                                                 t: the total number of testing subjects
                                                 h: the dimensions of the LR connectivity matrices
                        see train method above for model and args.
                Outputs:
                        for each epoch, prints out the mean testing MSE error
    ---------------------------------------------------------------------
    Copyright 2020 Megi Isallari, Istanbul Technical University.
    All rights reserved.
    """





#     parser = argparse.ArgumentParser(description='AGSR-Net')
#     parser.add_argument('--epochs', type=int, default=200, metavar='no_epochs',
#                         help='number of episode to train ')
#     parser.add_argument('--lr', type=float, default=0.0001, metavar='lr',
#                         help='learning rate (default: 0.0001 using Adam Optimizer)')
#     parser.add_argument('--lmbda', type=float, default=0.1, metavar='L',
#                         help='self-reconstruction error hyperparameter')
#     parser.add_argument('--lr_dim', type=int, default=160, metavar='N',
#                         help='adjacency matrix input dimensions')
#     parser.add_argument('--hr_dim', type=int, default=320, metavar='N',
#                         help='super-resolved adjacency matrix output dimensions')
#     parser.add_argument('--hidden_dim', type=int, default=320, metavar='N',
#                         help='hidden GraphConvolutional layer dimensions')
#     parser.add_argument('--padding', type=int, default=26, metavar='padding',
#                         help='dimensions of padding')
#     parser.add_argument('--mean_dense', type=float, default=0., metavar='mean',
#                         help='mean of the normal distribution in Dense Layer')
#     parser.add_argument('--std_dense', type=float, default=0.01, metavar='std',
#                         help='standard deviation of the normal distribution in Dense Layer')
#     parser.add_argument('--mean_gaussian', type=float, default=0., metavar='mean',
#                         help='mean of the normal distribution in Gaussian Noise Layer')
#     parser.add_argument('--std_gaussian', type=float, default=0.1, metavar='std',
#                         help='standard deviation of the normal distribution in Gaussian Noise Layer')
#     args = parser.parse_args()

import sys
sys.path.append("agsr_net") 
from agsr_net import preprocessing
from agsr_net.model import AGSRNet
from agsr_net.train import train, test
import argparse
from sklearn.model_selection import KFold
from MatrixVectorizer import MatrixVectorizer
import pandas as pd
import numpy as np
import torch
ks = [0.9, 0.7, 0.6, 0.5]

class Args:
    def __init__(self):
        self.epochs = 200  
        self.lr = 0.0001   
        self.lmbda = 0.1 
        self.lr_dim = 160     
        self.hr_dim = 320  
        self.hidden_dim = 320    
        self.padding = 26 
        self.mean_dense = 0.0    
        self.std_dense = 0.01      
        self.mean_gaussian = 0.0  
        self.std_gaussian = 0.1

ks = [0.9, 0.7, 0.6, 0.5]
args = Args() 




LR_TRAIN_DATA_PATH = "data/lr_train.csv"
LR_TEST_DATA_PATH = "data/lr_test.csv"
HR_TRAIN_DATA_PATH = "data/hr_train.csv"

df_lr_train = pd.read_csv(LR_TRAIN_DATA_PATH)
df_lr_test = pd.read_csv(LR_TEST_DATA_PATH)
df_hr_train = pd.read_csv(HR_TRAIN_DATA_PATH)

v_lr_train = np.zeros((len(df_lr_train), 160, 160))
v_lr_test = np.zeros((len(df_lr_test), 160, 160))
v_hr_train = np.zeros((len(df_hr_train), 268, 268))

mv = MatrixVectorizer()

for i, row in enumerate(df_lr_train.values):
    v_lr_train[i] = mv.anti_vectorize(row, 160)

for i, row in enumerate(df_lr_test.values):
    v_lr_test[i] = mv.anti_vectorize(row, 160)

for i, row in enumerate(df_hr_train.values):
    v_hr_train[i] = mv.anti_vectorize(row, 268)


k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

X = v_lr_train
y = v_hr_train

device = 'cuda' if torch.cuda.is_available() else 'cpu'

fold_metrics = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Split data for this fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
#     print(X_train.shape)    
#     print(y_train.shape)

#     print(type(X_train))

    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(device)

    # Before testing
    X_val = torch.tensor(X_val).to(device) 
    y_val = torch.tensor(y_val).to(device)
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AGSRNet(ks, args).to(device=device)

    train(model, X_train, y_train, args, device)
    test(model, X_val, y_val, args, device)


    # print(X_train.shape)
    # print(y_train.shape)
    # model = model(
    #         input_dim=12720,
    #         hidden_dims=[4096, 2048, 1024, 512],
    #         output_dim=35778,
    #         dropout_rate=0.3
    # )

    # model = AGSRNet(ks, args)

  
    

    # model, metrics = train.train(model, X_train, y_train, X_val, y_val)
    # fold_metrics.append(metrics)

    # print(f"  Validation Loss: {metrics['val_loss']:.4f}")
# print(fold_metrics)






# train(model, subjects_adj, subjects_ground_truth, args)
# test(model, test_adj, test_ground_truth, args)


