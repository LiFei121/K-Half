import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """
    Graph Convolutional Network for fact classification
    
    This is a local module file (GCN.py), not a third-party library.
    It implements a GCN model using DGL's GraphConv layers.
    """
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        """
        Args:
            g: DGL graph object
            in_feats: Input feature dimension
            n_hidden: Hidden layer dimension
            n_classes: Number of output classes
            n_layers: Number of GCN layers
            activation: Activation function (e.g., F.relu)
            dropout: Dropout rate
        """
        super(GCN, self).__init__()
        self.g = g
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        # Lazy import DGL to avoid import errors at module level
        try:
            from dgl.nn.pytorch import GraphConv
        except ImportError:
            try:
                from dgl.nn import GraphConv
            except ImportError as e:
                raise ImportError(
                    f"DGL library is required but could not be imported: {e}\n"
                    "Please install DGL with: pip install dgl\n"
                    "Or for CUDA support: pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html"
                )
        
        # Create GCN layers
        self.layers = nn.ModuleList()
        
        # Input layer
        if n_layers > 1:
            self.layers.append(GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))
        else:
            self.layers.append(GraphConv(in_feats, n_classes, allow_zero_in_degree=True))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
        
        # Output layer
        if n_layers > 1:
            self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
    
    def forward(self, features):
        """
        Forward pass
        
        Args:
            features: Node features tensor of shape (num_nodes, in_feats)
        
        Returns:
            Output logits of shape (num_nodes, n_classes)
        """
        h = features
        
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
            
            # Apply activation and dropout to all layers except the last one
            if i < len(self.layers) - 1:
                if self.activation is not None:
                    h = self.activation(h)
                h = self.dropout(h)
        
        return h

