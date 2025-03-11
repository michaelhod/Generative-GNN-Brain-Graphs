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


def compute_small_worldness(graph, seed=42):
    """
    Computes the small-worldness of a graph using an alternative method:
    
    1. Computes L (average shortest path length) and C (clustering coefficient).
    2. Generates a reference small-world graph using Watts-Strogatz model.
    3. Computes L_rand and C_rand from the reference graph.
    4. Computes small-worldness as:
    
       SW = (L / C) / (L_rand / C_rand)

    Returns:
        float: The small-worldness coefficient.
    """
    np.random.seed(seed)
    
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    
    avg_degree = int(2 * m / n)  # Ensuring even degree count
    
    try:
        L = nx.average_shortest_path_length(graph, weight="weight")
    except nx.NetworkXError:
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        L = nx.average_shortest_path_length(subgraph, weight="weight")

    C = nx.average_clustering(graph, weight="weight")
    
    if avg_degree < 2:  # Ensure valid degree for WS model
        avg_degree = 2
    small_world_ref = nx.watts_strogatz_graph(n, avg_degree, p=0.3, seed=seed)

    try:
        L_rand = nx.average_shortest_path_length(small_world_ref)
    except nx.NetworkXError:
        largest_cc_rand = max(nx.connected_components(small_world_ref), key=len)
        subgraph_rand = small_world_ref.subgraph(largest_cc_rand)
        L_rand = nx.average_shortest_path_length(subgraph_rand)

    C_rand = nx.average_clustering(small_world_ref)
    
    if C != 0 and C_rand != 0:
        sw = (L / C) / (L_rand / C_rand)
    else:
        sw = np.nan  

    return sw

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
    mae_sw = [] # small-worldness
    
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
        pred_sw = compute_small_worldness(pred_graph)

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")
        gt_cc = nx.clustering(gt_graph, weight="weight")
        gt_sw = compute_small_worldness(gt_graph)   


        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())
        pred_cc_values = list(pred_cc.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())
        gt_cc_values = list(gt_cc.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))
        mae_cc.append(mean_absolute_error(pred_cc_values, gt_cc_values))

            # Compute absolute difference for small-worldness
        if not np.isnan(pred_sw) and not np.isnan(gt_sw):
            mae_sw.append(abs(pred_sw - gt_sw))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrices[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt_matrices[i]))

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)
    avg_mae_cc = sum(mae_cc) / len(mae_cc)
    avg_mae_sw = sum(mae_sw) / len(mae_sw)

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
    print("Average MAE Small-World:", avg_mae_sw)

    data = {
        "MAE": mae,
        "PCC": pcc,
        "JS_Distance": js_dis,
        "MAE_BC": avg_mae_bc,
        "MAE_EC": avg_mae_ec,
        "MAE_PC": avg_mae_pc,
        "MAE_CC": avg_mae_cc,
        "MAE_SW": avg_mae_sw
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