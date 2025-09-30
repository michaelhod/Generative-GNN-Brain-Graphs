import sys
import time
sys.path.append('..')
import os
import hydra
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt

from src.train import train, eval, eval_test
from src.plot_utils import plot_adj_matrices
from src.dataset import load_dataset, load_test
from evaluation import evaluate_matrices
from MatrixVectorizer import MatrixVectorizer

def train_all_data(config, source_data, target_data):
    # Initialize results directory
    base_dir = config.experiment.base_dir
    model_name = 'stp_gsr_modified'
    dataset_type = config.dataset.name
    run_name = config.experiment.run_name
    run_dir = f'{base_dir}/{model_name}/final/'

    # Initialize folder structure for this run
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Train model for this fold
    train_output = train(config, 
                          source_data, 
                          target_data,
                          source_data,
                          target_data, 
                          run_dir)

    # Evaluate model for this fold
    eval_output, eval_loss = eval(config, 
                                  train_output['model'], 
                                  source_data, 
                                  target_data, 
                                  train_output['criterion'])

    # Final evaluation loss for this fold
    print(f"Final Validation Loss (Target): {eval_loss}")

    # Save source, taregt, and eval output for this fold
    np.save(f'{run_dir}/eval_output.npy', np.array(eval_output))
    np.save(f'{run_dir}/source.npy', np.array([s['mat'] for s in source_data]))
    np.save(f'{run_dir}/target.npy', np.array([t['mat'] for t in target_data]))

    # Evaluate predicted and target matrices
    predicted = np.array(eval_output)
    target = np.array([t['mat'] for t in target_data])

    return train_output

def eval_all_data(config, test_data):
    eval_output = eval_test(config, test_data)
    # find the shape of eval_output
    # Get upper triangular using MatrixVectorizer, eval_output = (112, 268,268)
    # for each eval_output[i], get the upper triangular part and convert it to a matrix
    save_to_csv(eval_output, 'prediction.csv')


def save_to_csv(eval_output, name):
    print(eval_output.shape)
    #return eval_output
    output = []
    for i in range(len(eval_output)):
        output.append(MatrixVectorizer.vectorize(eval_output[i], include_diagonal=False))
    eval_output = np.array(output)
    print(eval_output.shape)

    y_pred = eval_output.flatten()
    print(y_pred.shape)

    # Convert tensor to csv
    df = pd.DataFrame({
        'ID': np.arange(1, len(y_pred)+1),
        'Predicted': y_pred
    })

    df.to_csv(name, index=False)
    print('Prediction saved to {}'.format(name))

def generate_plots():
    print("Generating plots...")
    df_fold_0 = pd.read_csv('../evaluation/soap/fold_0.csv')
    df_fold_1 = pd.read_csv('../evaluation/soap/fold_1.csv')
    df_fold_2 = pd.read_csv('../evaluation/soap/fold_2.csv')

    # Calculate the mean and standard deviation
    df_all = pd.concat([df_fold_0, df_fold_1, df_fold_2])
    df_average = df_all.groupby(level=0).mean()
    df_std = df_all.groupby(level=0).std()

    # Ensure column names match when using error bars
    df_std.columns = df_average.columns  # Align std columns with mean columns

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot Fold 1
    df_fold_0.iloc[0].plot(ax=axes[0, 0], title='Fold 1', kind='bar', color=colors, fontsize=12)

    # Plot Fold 2
    df_fold_1.iloc[0].plot(ax=axes[0, 1], title='Fold 2', kind='bar', color=colors, fontsize=12)

    # Plot Fold 3
    df_fold_2.iloc[0].plot(ax=axes[1, 0], title='Fold 3', kind='bar', color=colors, fontsize=12)

    # Plot Average with Standard Deviation Error Bars
    df_average.iloc[0].plot(ax=axes[1, 1], title='Average', kind='bar', color=colors, fontsize=12, 
                            yerr=df_std.iloc[0].values, capsize=5, alpha=0.7)

    plt.tight_layout()

    # Save the plot
    plt.savefig('../evaluation/soap/evaluation.png')



@hydra.main(version_base="1.3.2", config_path="configs", config_name="experiment")
def main(config):
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    # Sleep for 20 seconds to allow for memory logging
    
    #print("Sleeping for 20 seconds to allow for memory logging")
    #time.sleep(20)


    if config.experiment.output_csv:
        print("Training on all data and outputting to csv")
        # First train on all data
        source_data, target_data = load_dataset(config)
        train_all_data(config, source_data, target_data)

        # Then evaluate on test data and save to csv
        test_data = load_test(config)
        eval_all_data(config, test_data)
        return

    print("Starting k-fold cross validation")

    kf = KFold(n_splits=config.experiment.kfold.n_splits, 
               shuffle=config.experiment.kfold.shuffle, 
               random_state=config.experiment.kfold.random_state)

    # Initialize folder structure for this run
    base_dir = config.experiment.base_dir
    model_name = config.model.name
    dataset_type = config.dataset.name
    run_name = config.experiment.run_name
    run_dir = f'{base_dir}/{model_name}/{dataset_type}/{run_name}/'

    # Load dataset
    source_data, target_data = load_dataset(config)


    for fold, (train_idx, val_idx) in enumerate(kf.split(source_data)):
        print(f"Training Fold {fold+1}/3")

        # Initialize results directory
        res_dir = f'{run_dir}fold_{fold+1}/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Fetch training and val data for this fold
        source_data_train = [source_data[i] for i in train_idx]
        target_data_train = [target_data[i] for i in train_idx]
        source_data_val = [source_data[i] for i in val_idx]
        target_data_val = [target_data[i] for i in val_idx]

        # Train model for this fold
        train_output = train(config, 
                              source_data_train, 
                              target_data_train,
                              source_data_val,
                              target_data_val, 
                              res_dir)

        # Evaluate model for this fold
        eval_output, eval_loss = eval(config, 
                                      train_output['model'], 
                                      source_data_val, 
                                      target_data_val, 
                                      train_output['criterion'])
        
        # Save predictions for this fold to csv
        save_to_csv(np.array(eval_output), f'{res_dir}/predictions_fold{fold+1}.csv')

        # Final evaluation loss for this fold
        print(f"Final Validation Loss (Target): {eval_loss}")

        # Save source, taregt, and eval output for this fold
        np.save(f'{res_dir}/eval_output.npy', np.array(eval_output))
        np.save(f'{res_dir}/source.npy', np.array([s['mat'] for s in source_data_val]))
        np.save(f'{res_dir}/target.npy', np.array([t['mat'] for t in target_data_val]))

        # Evaluate predicted and target matrices
        predicted = np.array(eval_output)
        target = np.array([t['mat'] for t in target_data_val])
        evaluate_matrices(predicted, target, fold_num=fold, model_name='soap', all_metrics=True)

    # Generate plots
    generate_plots()

if __name__ == "__main__":
    main()
