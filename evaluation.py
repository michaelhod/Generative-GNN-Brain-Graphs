from matplotlib.pylab import eigvalsh
from MatrixVectorizer import MatrixVectorizer

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import torch
import networkx as nx
import numpy as np
import torch
import pandas as pd

def compute_laplacian_energy(graph):
    """
    Compute the Laplacian energy of a graph.
    
    Params:
    - graph : networkx.Graph
        The graph for which to compute the Laplacian energy.
    
    Returns:
    - float: The Laplacian energy of the graph.
    """
    # Compute Laplacian matrix
    L = nx.laplacian_matrix(graph).toarray()
    
    # Compute eigenvalues of the Laplacian matrix
    eigenvalues = eigvalsh(L)
    
    # Number of nodes
    n = graph.number_of_nodes()
    
    # Number of edges
    m = graph.number_of_edges()
    
    # Compute Laplacian Energy
    avg_lambda = (2 * m) / n
    energy = np.sum(np.abs(eigenvalues - avg_lambda))

    return energy
def compute_ricci_curvature(graph):
    """
    Compute the Ricci curvature of a graph.
    
    Params:
    - graph : networkx.Graph
        The graph for which to compute the Ricci curvature.
    
    Returns:
    - dict: A dictionary containing the Ricci curvature values for each edge in the graph.
    """
    # Compute Ricci curvature
    orc = OllivierRicci(graph, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()

    return orc.G.edges(data="ricciCurvature")
    return energy

def evaluate_matrices(pred_matrices, gt_matrices, fold_num, model_name, all_metrics=False):
    """
    Evaluate the predicted matrices against the ground-truth matrices.

    Params:
    - pred_matrices : numpy.ndarray
        Predicted adjacency matrices of shape (n_samples, adj_size, adj_size).
    - gt_matrices : numpy.ndarray
        Ground truth adjacency matrices of shape (n_samples, adj_size, adj_size).
    - all_metrics : bool, optional
        Flag to compute all metrics. Defaults to False - only computes MAE.
    
    Returns:
    - dict: A dictionary containing the evaluation metrics:
        - MAE: Mean Absolute Error between flattened matrices
        If all_metrics=True, also includes:
        - PCC: Pearson Correlation Coefficient
        - JS_Distance: Jensen-Shannon Distance
        - MAE_BC: Average Mean Absolute Error for betweenness centrality
        - MAE_EC: Average Mean Absolute Error for eigenvector centrality
        - MAE_PC: Average Mean Absolute Error for PageRank centrality
    """
    # post-processing
    pred_matrices[pred_matrices < 0] = 0

    # flattened matrices
    pred_1d_list = []
    gt_1d_list = []
    results = {}
    
    num_test_samples = pred_matrices.shape[0]

    # If only calculating MAE, we can just flatten and compute directly
    if not all_metrics:
        # Vectorize and concatenate all matrices at once
        for i in range(num_test_samples):
            pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrices[i]))
            gt_1d_list.append(MatrixVectorizer.vectorize(gt_matrices[i]))
            
        pred_1d = np.concatenate(pred_1d_list)
        gt_1d = np.concatenate(gt_1d_list)
        
        # Compute MAE only
        mae = mean_absolute_error(pred_1d, gt_1d)
        print("MAE: ", mae)
        
        return {"MAE": mae}
    
    # If all_metrics is True, calculate all the centrality metrics too
    mae_bc = []
    mae_ec = []
    mae_pc = []
    mae_cc = [] # clustering coefficient
    mae_laplacian = [] # Laplacian energy
    
    # Iterate over each test sample
    for i in range(num_test_samples):
        print(f"Processing sample {i+1}/{num_test_samples}")
        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(pred_matrices[i], edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_matrices[i], edge_attr="weight")

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")
        pred_cc = nx.clustering(pred_graph, weight="weight")
        pred_laplacian = compute_laplacian_energy(pred_graph)

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")
        gt_cc = nx.clustering(gt_graph, weight="weight")
        gt_laplacian = compute_laplacian_energy(gt_graph)   


        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())
        pred_cc_values = list(pred_cc.values())
        #pred_laplacian_values = list(pred_laplacian.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())
        gt_cc_values = list(gt_cc.values())
        #gt_laplacian_values = list(gt_laplacian.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))
        mae_cc.append(mean_absolute_error(pred_cc_values, gt_cc_values))
        mae_laplacian.append(mean_absolute_error([pred_laplacian], [gt_laplacian]))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrices[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt_matrices[i]))

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)
    avg_mae_cc = sum(mae_cc) / len(mae_cc)
    avg_mae_laplacian = sum(mae_laplacian) / len(mae_laplacian)

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    # Compute metrics
    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    print("MAE: ", mae)
    print("PCC: ", pcc)
    print("Jensen-Shannon Distance: ", js_dis)
    print("Average MAE betweenness centrality:", avg_mae_bc)
    print("Average MAE eigenvector centrality:", avg_mae_ec)
    print("Average MAE PageRank centrality:", avg_mae_pc)
    print("Average MAE clustering coefficient:", avg_mae_cc)
    print("Average MAE Laplacian:", avg_mae_laplacian)

    data = {
        "MAE": mae,
        "PCC": pcc,
        "JS_Distance": js_dis,
        "MAE_BC": avg_mae_bc,
        "MAE_EC": avg_mae_ec,
        "MAE_PC": avg_mae_pc,
        "MAE_CC": avg_mae_cc,
        "MAE_Laplacian": avg_mae_laplacian
    }

    df = pd.DataFrame(data=data, index=[0])

    df.to_csv(f"../evaluation/{model_name}/fold_{fold_num}.csv", index=False)

    return data

if __name__ == "__main__":    
    # Example data generation (just for demonstration)
    num_test_samples = 20
    num_roi = 10
    
    # Create random model output
    pred_matrices = torch.randn(num_test_samples, num_roi, num_roi).numpy()
    
    # Create random ground-truth data
    gt_matrices = torch.randn(num_test_samples, num_roi, num_roi).numpy()
    gt_matrices[gt_matrices < 0] = 0  # Pre-process ground truth
    
    # print shapes
    print(pred_matrices.shape)
    print(gt_matrices.shape)

    # Evaluate
    results = evaluate_matrices(pred_matrices, gt_matrices)
    
    # Print results
    for metric, value in results.items():
        print(f"{metric}: {value}")