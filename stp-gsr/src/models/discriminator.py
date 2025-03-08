

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Discriminator(nn.Module):
#     def __init__(self, n_nodes, hidden_dim):
#         super(Discriminator, self).__init__()
#         self.n_nodes = n_nodes
#         self.hidden_dim = hidden_dim

#         self.fc1 = nn.Linear(n_nodes * n_nodes, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc3 = nn.Linear(hidden_dim // 2, 1)
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, adj):
#         x = adj.view(-1, self.n_nodes * self.n_nodes)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = torch.sigmoid(self.fc3(x))

#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, n_nodes, hidden_dim):
        """
        Discriminator for graph super-resolution
        
        Args:
            n_nodes (int): Number of nodes in the high-resolution graph (target)
            hidden_dim (int): Dimension of hidden layers
        """
        super(Discriminator, self).__init__()
        self.n_nodes = n_nodes  # This should be the HR size (268)
        self.hidden_dim = hidden_dim
        self.input_size = n_nodes * n_nodes  # For a 268x268 adjacency matrix: 71,824
        
        # Define network layers
        self.fc1 = nn.Linear(self.input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
        
        print(f"Discriminator initialized with n_nodes={n_nodes}, input_size={self.input_size}")
        
        # Initialize weights for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, adj):
        """
        Forward pass through the discriminator
        
        Args:
            adj (torch.Tensor): Adjacency matrix of shape [batch_size, n_nodes, n_nodes]
                               or [n_nodes, n_nodes]
        
        Returns:
            torch.Tensor: Probability that the input is real (1) or fake (0)
        """
        # Store original shape for debugging
        orig_shape = adj.shape
        
        # Handle different input dimensions
        if adj.dim() == 3:  # [batch_size, n_nodes, n_nodes]
            batch_size = adj.size(0)
            # Reshape to [batch_size, n_nodes*n_nodes]
            x = adj.reshape(batch_size, -1)
        elif adj.dim() == 2:  # [n_nodes, n_nodes]
            # Add batch dimension and reshape to [1, n_nodes*n_nodes]
            x = adj.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected input shape: {adj.shape}, expected 2D or 3D tensor")
        
        # Check dimensions
        if x.size(1) != self.input_size:
            expected_nodes = int(self.input_size ** 0.5)
            actual_nodes = int(x.size(1) ** 0.5) if x.size(1) > 0 else 0
            raise ValueError(
                f"Input dimension mismatch. Expected {self.n_nodes}x{self.n_nodes} adjacency matrix "
                f"({self.input_size} elements), but got shape {orig_shape} with {x.size(1)} elements "
                f"(equivalent to about {actual_nodes}x{actual_nodes})."
            )
        
        # Forward pass
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        # Remove singleton dimensions if batch_size=1
        return x.squeeze()