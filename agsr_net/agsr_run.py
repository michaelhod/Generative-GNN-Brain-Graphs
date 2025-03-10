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
from model import AGSRNet
from train import train, test
import argparse
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch
sys.path.append('..')
from MatrixVectorizer import MatrixVectorizer
from evaluation import evaluate_matrices

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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_model = None
best_score = float('inf')
fold_metrics = []
model = AGSRNet(ks, args)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Split data for this fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
#     print(X_train.shape)    
#     print(y_train.shape)

#     print(type(X_train))


    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AGSRNet(ks, args)

    
    all_metrics = True

    train(model, X_train, y_train, args)
    preds_list, ground_truth = test(model, X_val, y_val, args)

    metrics = evaluate_matrices(preds_list, ground_truth, all_metrics=all_metrics)
    
    fold_metrics.append(metrics)
    
    print("=== Evaluation Results for Fold {} ===".format(fold + 1))
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    if not all_metrics:
        print(f"MAE: {metrics['MAE']:.6f}")
    else:
        print(f"MAE: {metrics['MAE']:.6f}")
        print(f"PCC: {metrics['PCC']:.6f}")
        print(f"Jensen-Shannon Distance: {metrics['JS_Distance']:.6f}")
        print(f"Average MAE betweenness centrality: {metrics['MAE_BC']:.6f}")
        print(f"Average MAE eigenvector centrality: {metrics['MAE_EC']:.6f}")
        print(f"Average MAE PageRank centrality: {metrics['MAE_PC']:.6f}")
    
    # Update best model if this fold is better
    
    print(f"MAE: {metrics['MAE']:.6f}")

    if metrics['MAE'] < best_score:
        best_score = metrics['MAE']
        best_model = model.state_dict()

    break

avg_metrics = {}
for metric in fold_metrics[0].keys():
    avg_metrics[metric] = sum(fold[metric] for fold in fold_metrics) / len(fold_metrics)

print("\n=== Average Metrics Across All Folds ===")
for metric, value in avg_metrics.items():
    print(f"Average {metric}: {value:.6f}")

print(f"\nAverage MAE: {avg_metrics['MAE']:.6f}")

# Save best model after all folds
if best_model is not None:
    torch.save(best_model, 'best_model.pt')
    print(f"\nBest model saved with MAE: {best_score:.6f}")


    # print(X_train.shape)
    # print(y_train.shape)
    # model = model(
    #         input_dim=12720,
    #         hidden_dims=[4096, 2048, 1024, 512],
    #         output_dim=35778,
    #         dropout_rate=0.3
    # )

    # model = AGSRNet(ks, args)

lr_dim, hr_dim = 160, 268


preds_list = []
for lr in v_lr_test:
    all_zeros_lr = not np.any(lr)
    if all_zeros_lr == False:
        lr = torch.from_numpy(lr).type(torch.FloatTensor)
    
        final_preds, gold = model(lr, hr_dim, hr_dim)
        preds_list.append(final_preds.detach().numpy())
    
preds_list = np.array(preds_list)
melted_preds = preds_list.flatten()

submission_df = pd.DataFrame({
    'ID': range(1, len(melted_preds) + 1),  # IDs from 1 to 4,007,136
    'Predicted': melted_preds
})

submission_df.to_csv('submission.csv', index=False)



    # model, metrics = train.train(model, X_train, y_train, X_val, y_val)
    # fold_metrics.append(metrics)

    # print(f"  Validation Loss: {metrics['val_loss']:.4f}")
# print(fold_metrics)






# train(model, subjects_adj, subjects_ground_truth, args)
# test(model, test_adj, test_ground_truth, args)


