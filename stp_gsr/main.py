import sys
sys.path.append('..')
import os
import hydra
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd

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

    df.to_csv('prediction.csv', index=False)
    print('Prediction saved to prediction.csv')



@hydra.main(version_base="1.3.2", config_path="configs", config_name="experiment")
def main(config):
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

            

    train_all = False
    if train_all:
        # Load dataset
        source_data, target_data = load_dataset(config)
        train_all_data(config, source_data, target_data)

    do_test = False
    if do_test:
        test_data = load_test(config)
        eval_all_data(config, test_data)
        return


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


        # Plot predictions for a random sample
        idx = 6
        source_mat_test = source_data_val[idx]['mat']
        target_mat_test = target_data_val[idx]['mat']
        eval_output_t = eval_output[idx]

        plot_adj_matrices(source_mat_test, 
                          target_mat_test, 
                          eval_output_t, 
                          idx, 
                          res_dir, 
                          file_name=f'eval_sample{idx}')
        

if __name__ == "__main__":
    main()
