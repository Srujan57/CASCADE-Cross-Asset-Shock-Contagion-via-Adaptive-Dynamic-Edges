"""
models/evolvegcn.py

EvolveGCN-H: Temporal Graph Neural Network for cross-asset contagion modeling.

Key fix over naive implementation:
    Using .data to inject evolved W into GCNConv breaks the computation graph —
    gradients stop at the GCN weight and never reach the GRU or W parameters.
    
    Solution: custom GCNLayer that accepts W as a forward() argument, keeping
    the entire chain (loss → GCN → W_evolved → GRU → W_stored) in the graph.

Architecture:
    Nodes:  10 asset classes (BTC, ETH, SPY, EEM, LQD, HYG, TLT, GLD, USO, DXY)
    Edges:  DCC-GARCH dynamic correlations (weekly snapshots)
    Output: predicted return at t+1, t+5, t+10 for all nodes

Reference: Pareja et al. "EvolveGCN: Evolving Graph Convolutional Networks
           for Dynamic Graphs." AAAI 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


# ─────────────────────────────────────────────────────────────────────────────
# Custom GCN Layer
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(MessagePassing):
    """
    A single graph convolution layer that accepts W as an argument.

    Why not just use GCNConv?
        GCNConv stores W internally. To inject an evolved W you'd have to use
        .data assignment, which detaches from the computation graph and kills
        gradient flow back through the GRU. By accepting W as a forward()
        argument, it stays inside the graph — PyTorch can trace all the way
        from the loss back through the GCN, through W_evolved, through the
        GRU, to the stored W parameters.

    What this layer does (standard GCN formula):
        1. Add self-loops to the graph (each node aggregates itself too)
        2. Compute symmetric normalization: D^{-1/2} * A * D^{-1/2}
           (prevents high-degree nodes from dominating aggregation)
        3. Transform node features: X_new = A_norm @ X @ W^T + bias
        4. Return updated node representations
    """

    def __init__(self, in_channels, out_channels):
        # aggr='add' means: for each node, sum up messages from all neighbors
        super().__init__(aggr='add')
        # Bias is still a stored parameter — only W is passed externally
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_weight, W):
        """
        Args:
            x           : node features, shape (num_nodes, in_channels)
            edge_index  : connectivity, shape (2, num_edges)
            edge_weight : correlation strengths, shape (num_edges,)
            W           : evolved weight matrix from GRU, shape (out, in)
                          Passed as argument — stays in computation graph.
        Returns:
            Updated node features, shape (num_nodes, out_channels)
        """
        num_nodes = x.size(0)

        # ── Step 1: Add self-loops ────────────────────────────────────────
        # Without self-loops, a node's own features don't contribute to its
        # updated representation — only its neighbors do. Self-loops fix that.
        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=num_nodes
        )

        # ── Step 2: Symmetric normalization ──────────────────────────────
        # Compute degree of each node (sum of edge weights)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype)

        # D^{-1/2}: inverse square root of degree
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # handle isolated nodes

        # Normalize each edge: d_i^{-1/2} * w_ij * d_j^{-1/2}
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # ── Step 3: Linear transform with evolved W ───────────────────────
        # F.linear(x, W, bias) computes x @ W.T + bias
        # W is the GRU's output — still connected to the computation graph.
        # This is what allows gradients to flow back to the GRU.
        x = F.linear(x, W, self.bias)

        # ── Step 4: Propagate (message passing) ───────────────────────────
        # For each node: aggregate transformed features from neighbors,
        # weighted by the normalized edge weights.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """
        Called internally by propagate() for each edge.
        x_j  : feature of the source node for this edge
        norm : normalized edge weight
        Returns the message to be aggregated at the target node.
        """
        return norm.view(-1, 1) * x_j


# ─────────────────────────────────────────────────────────────────────────────
# EvolveGCN-H
# ─────────────────────────────────────────────────────────────────────────────

class EvolveGCNH(nn.Module):
    """
    EvolveGCN-H: a GRU evolves the GCN weight matrix W at each timestep.

    The "H" variant uses W itself as both the GRU input and hidden state.
    This means the model's internal transformation parameters adapt over time,
    allowing it to learn different contagion patterns in crisis vs calm regimes.

    Args:
        node_features : number of input features per node (4 in this project)
        hidden_dim    : size of internal representations (64 from config.yaml)
        num_layers    : number of stacked GCN layers (2 from config.yaml)
        dropout       : dropout rate (0.3 from config.yaml)
    """

    def __init__(self, node_features, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()

        self.num_layers    = num_layers
        self.hidden_dim    = hidden_dim
        self.node_features = node_features
        self.dropout       = nn.Dropout(dropout)

        self.convs    = nn.ModuleList()
        self.grus     = nn.ModuleList()
        self.w_shapes = []

        for i in range(num_layers):
            in_dim = node_features if i == 0 else hidden_dim
            self.convs.append(GCNLayer(in_dim, hidden_dim))

            w_shape = (hidden_dim, in_dim)
            self.w_shapes.append(w_shape)

            flat_size = hidden_dim * in_dim
            self.grus.append(nn.GRUCell(flat_size, flat_size))

        # Stored W matrices — starting point, evolved by GRU each timestep
        self.W = nn.ParameterList([
            nn.Parameter(torch.zeros(shape))
            for shape in self.w_shapes
        ])

        # Three prediction heads — one per forecasting horizon
        self.predictors = nn.ModuleDict({
            "t1":  self._make_predictor(hidden_dim),
            "t5":  self._make_predictor(hidden_dim),
            "t10": self._make_predictor(hidden_dim),
        })

    def _make_predictor(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, snapshots, return_states=False):
        """
        Args:
            snapshots     : list of (x, edge_index, edge_weight) tuples
            return_states : if True, also return W matrices and node embeddings
                            per timestep — used in evaluate.py for regime analysis
                            and contagion map generation

        Returns:
            predictions : dict {"t1", "t5", "t10"}, each (T, N, 1)
            states      : (only if return_states=True)
                          dict with:
                            "w_norms"    : (T, num_layers) — Frobenius norm of W per layer
                            "embeddings" : (T, N, hidden_dim) — final node embeddings
        """

        preds      = {"t1": [], "t5": [], "t10": []}
        w_norms    = []   # Frobenius norm of each layer's W per timestep
        embeddings = []   # final node embeddings per timestep

        W_current = [w.clone() for w in self.W]

        for x, edge_index, edge_weight in snapshots:

            h              = x
            timestep_norms = []

            for i in range(self.num_layers):

                W_flat       = W_current[i].view(1, -1)
                W_evolved    = self.grus[i](W_flat, W_flat)
                W_current[i] = W_evolved.view(self.w_shapes[i])

                if return_states:
                    # Frobenius norm = sqrt(sum of squared elements)
                    # Measures how "active" the weight matrix is at this timestep
                    norm = W_current[i].detach().norm(p='fro').item()
                    timestep_norms.append(norm)

                h = self.convs[i](h, edge_index, edge_weight, W_current[i])
                h = torch.relu(h)
                h = self.dropout(h)

            for horizon, predictor in self.predictors.items():
                preds[horizon].append(predictor(h))

            if return_states:
                w_norms.append(timestep_norms)
                embeddings.append(h.detach())   # (N, hidden_dim)

        predictions = {k: torch.stack(v, dim=0) for k, v in preds.items()}

        if return_states:
            states = {
                "w_norms":    torch.tensor(w_norms, dtype=torch.float),  # (T, L)
                "embeddings": torch.stack(embeddings, dim=0),            # (T, N, H)
            }
            return predictions, states

        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check — run: python models/evolvegcn.py
# Expected: all shapes PASS, backward PASS, all params grad=YES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    torch.manual_seed(42)

    NUM_NODES     = 10
    NODE_FEATURES = 4
    HIDDEN_DIM    = 64
    NUM_LAYERS    = 2
    DROPOUT       = 0.3
    NUM_SNAPSHOTS = 20

    model = EvolveGCNH(
        node_features=NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    print("Model architecture:")
    print(model)
    print(f"\nTotal trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def make_fake_snapshot(num_nodes, node_features, num_edges=25):
        x           = torch.randn(num_nodes, node_features)
        src         = torch.randint(0, num_nodes, (num_edges,))
        dst         = torch.randint(0, num_nodes, (num_edges,))
        edge_index  = torch.stack([src, dst], dim=0)
        edge_weight = torch.rand(num_edges)
        return x, edge_index, edge_weight

    snapshots = [make_fake_snapshot(NUM_NODES, NODE_FEATURES)
                 for _ in range(NUM_SNAPSHOTS)]

    model.train()
    predictions = model(snapshots)

    print("\nOutput shapes (should be [num_snapshots, num_nodes, 1]):")
    for horizon, pred in predictions.items():
        status = "PASS" if pred.shape == (NUM_SNAPSHOTS, NUM_NODES, 1) else "FAIL"
        print(f"  {horizon}: {str(list(pred.shape)):20s} — {status}")

    fake_loss = sum(v.sum() for v in predictions.values())
    fake_loss.backward()
    print("\nBackward pass: PASS — gradients computed successfully")

    print("\nGradient check:")
    all_pass = True
    for name, param in model.named_parameters():
        has_grad = param.grad is not None
        if not has_grad:
            all_pass = False
        print(f"  {name:45s} grad={'YES' if has_grad else 'NO  ← PROBLEM'}")

    print(f"\nAll gradients flowing: {'YES — model correctly wired' if all_pass else 'NO — check above'}")