import sys
sys.path.append("../..")
import torch
import numpy as np
import networkx as nx
from functools import partial
from torch_geometric.data import Data
import os
import pandas as pd
from MatrixVectorizer import MatrixVectorizer


def load_dataset(config):
    n_source_nodes = config.dataset.n_source_nodes
    n_target_nodes = config.dataset.n_target_nodes
    n_samples = config.dataset.n_samples
    
    if config.dataset.name == 'custom':
        # Import adjacency matrices from file
        LR_TRAIN_DATA_FILE_NAME = "lr_train.csv"
        #LR_TEST_DATA_FILE_NAME = "lr_test.csv"
        HR_TRAIN_DATA_FILE_NAME = "hr_train.csv"

        DATA_DIR = os.path.join("..", "data")

        LR_TRAIN_DATA_PATH = os.path.join(DATA_DIR, LR_TRAIN_DATA_FILE_NAME)
        HR_TRAIN_DATA_PATH = os.path.join(DATA_DIR, HR_TRAIN_DATA_FILE_NAME)

        df_lr_train = pd.read_csv(LR_TRAIN_DATA_PATH)
        df_hr_train = pd.read_csv(HR_TRAIN_DATA_PATH)

        source_mat_all = create_custom_graph(df_lr_train, config.dataset.n_source_nodes) # the csv file already contains all of our graphs
        target_mat_all = create_custom_graph(df_hr_train, config.dataset.n_target_nodes)

    else:
        raise ValueError(f"Unsupported dataset type: {config.dataset.name}")
    
    # Convert to torch tensors
    source_mat_all = [torch.tensor(x, dtype=torch.float) for x in source_mat_all]
    target_mat_all = [torch.tensor(x, dtype=torch.float) for x in target_mat_all]
        
    # Convert to PyG
    node_feat_init = config.dataset.node_feat_init
    node_feat_dim = config.dataset.node_feat_dim
    pyg_partial = partial(create_pyg_graph, node_feature_init=node_feat_init, node_feat_dim=node_feat_dim)
    
    source_pyg_all = [pyg_partial(x, n_source_nodes) for x in source_mat_all]
    target_pyg_all = [pyg_partial(x, n_target_nodes) for x in target_mat_all]

    # Prepare source and target data
    source_data = [{'pyg': source_pyg, 'mat': source_mat} for source_pyg, source_mat in zip(source_pyg_all, source_mat_all)]
    target_data = [{'pyg': target_pyg, 'mat': target_mat} for target_pyg, target_mat in zip(target_pyg_all, target_mat_all)]

    return source_data, target_data

def create_custom_graph(df: pd.DataFrame, node_size: int):
    # df is a pandas dataframe that contains all of our graphs
    # Convert each row to a matrix using the vectorizer
    matrix_size = node_size
    
    df_matrices = []
    for i in range(df.shape[0]):
        row_data = df.iloc[i].values
        matrix = MatrixVectorizer.anti_vectorize(row_data, matrix_size=matrix_size, include_diagonal=False)
        df_matrices.append(matrix)
    
    return np.array(df_matrices)


def create_pyg_graph(x, n_nodes, node_feature_init='adj', node_feat_dim=1):
    """
    Create a PyTorch Geometric graph data object from given adjacency matrix.
    """
    # Initialise edge features
    if isinstance(x, torch.Tensor):
        edge_attr = x.view(-1, 1)
    else:
        edge_attr = torch.tensor(x, dtype=torch.float).view(-1, 1)


    # Initialise node features
    # From adjacency matrix
    if node_feature_init == 'adj':
        if isinstance(x, torch.Tensor):
            # node_feat = x.clone().detach()
            node_feat = x
        else:
            node_feat = torch.tensor(x, dtype=torch.float)

    # Random initialisation
    elif node_feature_init == 'random':
        node_feat = torch.randn(n_nodes, node_feat_dim, device=edge_attr.device)

    # Ones initialisation
    elif node_feature_init == 'ones':
        node_feat = torch.ones(n_nodes, node_feat_dim, device=edge_attr.device)

    else:
        raise ValueError(f"Unsupported node feature initialization: {node_feature_init}")


    rows, cols = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
    pos_edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

    pyg_graph = Data(x=node_feat, pos_edge_index=pos_edge_index, edge_attr=edge_attr)
    
    return pyg_graph
