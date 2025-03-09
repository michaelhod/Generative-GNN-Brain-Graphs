import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm, GATConv

from src.dual_graph_utils import create_dual_graph, create_dual_graph_feature_matrix
 
class TargetEdgeInitializer(nn.Module):
    """TransformerConv based taregt edge initialization model"""
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, hidden_dim=64, num_layers=2, edge_dim=1, 
                 dropout=0.2, beta=False):
        super().__init__()
        assert n_target_nodes % num_heads == 0

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.phi = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(2*n_target_nodes, 1)))

        if num_layers == 1:
            self.convs.append(TransformerConv(n_source_nodes, n_target_nodes // num_heads, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta))
            self.bns.append(GraphNorm(n_target_nodes))
        else:
            self.convs.append(TransformerConv(n_source_nodes, hidden_dim // num_heads, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta))
            self.bns.append(GraphNorm(hidden_dim))

            for _ in range(num_layers - 2):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim // num_heads, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta))
                self.bns.append(GraphNorm(hidden_dim))

            self.convs.append(TransformerConv(hidden_dim, n_target_nodes // num_heads, heads=num_heads,
                                              dropout=dropout, edge_dim=edge_dim, beta=beta))
            self.bns.append(GraphNorm(n_target_nodes)) 

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # Update node embeddings for the source graph

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)

        # Super-resolve source graph using matrix multiplication
        xt = x.T @ x    # xt will be treated as the adjacency matrix of the target graph

        # Normalize values to be between [0, 1]
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        # Fetch and reshape upper triangular part to get dual graph's node feature matrix
        ut_mask = torch.triu(torch.ones_like(xt), diagonal=1).bool()
        x = torch.masked_select(xt, ut_mask).view(-1, 1)

        return x
    

class DualGraphLearner(nn.Module):
    """Update node features of the dual graph"""
    def __init__(self, in_dim, out_dim=1, num_heads=1,
                 dropout=0.2, beta=False):
        super().__init__()

        # Here, we override num_heads to be 1 since we output scalar primal edge weights
        # In future work, we can experiment with multiple heads

        self.conv1 = TransformerConv(in_dim, out_dim, 
                                     heads=num_heads,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(out_dim)

    def forward(self, x, edge_index):
        # Update embeddings for the dual nodes/ primal edges 
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        xt = F.relu(x)
        
        # Normalize values to be between [0, 1]
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        return xt
    

class STPGSR(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_source_nodes = config.dataset.n_source_nodes
        n_target_nodes = config.dataset.n_target_nodes

        self.target_edge_initializer = TargetEdgeInitializer(
                            n_source_nodes,
                            n_target_nodes,
                            num_heads=config.model.target_edge_initializer.num_heads,
                            hidden_dim=config.model.target_edge_initializer.hidden_dim,
                            num_layers=config.model.target_edge_initializer.num_layers,
                            edge_dim=config.model.target_edge_initializer.edge_dim,
                            dropout=config.model.target_edge_initializer.dropout,
                            beta=config.model.target_edge_initializer.beta
        )
        self.dual_learner = DualGraphLearner(
                            in_dim=config.model.dual_learner.in_dim,
                            out_dim=config.model.dual_learner.out_dim,
                            num_heads=config.model.dual_learner.num_heads,
                            dropout=config.model.dual_learner.dropout,
                            beta=config.model.dual_learner.beta
        )
        print(self.target_edge_initializer)
        print("-"*50)
        print(self.dual_learner)
        # Create dual graph domain: Assume a fully connected simple graph
        fully_connected_mat = torch.ones((n_target_nodes, n_target_nodes), dtype=torch.float)   # (n_t, n_t)
        self.dual_edge_index, _ = create_dual_graph(fully_connected_mat)    # (2, n_t*(n_t-1)/2), (n_t*(n_t-1)/2, 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dual_edge_index = self.dual_edge_index.to(self.device)


    def forward(self, source_pyg, target_mat):
        source_pyg = source_pyg.to(self.device)
        target_mat = target_mat.to(self.device)
        
        # Initialize target edges
        target_edge_init = self.target_edge_initializer(source_pyg)
        # Update target edges in the dual space 
        dual_pred_x = self.dual_learner(target_edge_init, self.dual_edge_index)

        # Convert target matrix into edge feature matrix
        dual_target_x = create_dual_graph_feature_matrix(target_mat)

        return dual_pred_x, dual_target_x
