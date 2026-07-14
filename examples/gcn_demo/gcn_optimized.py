import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

from src.models.base import BaseModel


num_atom_type = 119
num_chirality_tag = 4


class SimpleGCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleGCNConv, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        deg.scatter_add_(
            0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device)
        )
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.linear(x)

        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.view(-1, 1))

        return out


# EVOLVE-BLOCK-START
class GCN(BaseModel):
    def __init__(
        self,
        in_feat: int = 113,
        hidden_feat: int = 64,
        out_feat: int = 32,
        out: int = 1,
        grid_feat: int = 1,
        num_layers: int = 5,
        pooling: str = "mean",
        use_bias: bool = False,
        drop_ratio: float = 0.0,
        feat_dim: int = 256,
    ):
        super(GCN, self).__init__()
        self.num_layer = num_layers
        self.emb_dim = hidden_feat * 2
        self.drop_ratio = drop_ratio
        self.pooling = pooling

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GCNConv(self.emb_dim, self.emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        if pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool

        # JK representation size: 3 (mean, max, sum) * emb_dim * 3 layers selected (initial, middle, last)
        self.jk_proj = nn.Linear(self.emb_dim * 3 * 3, self.emb_dim * 2)

        # Use a combination of mean and max pooling to capture diverse graph properties
        self.feat_lin = nn.Sequential(
            nn.Linear(self.emb_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU()
        )

        # A deeper prediction head with LayerNorm and residual connection to stabilize learning
        self.pred_head1 = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(drop_ratio),
        )
        self.pred_head2 = nn.Linear(feat_dim, out)

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            h = F.gelu(h)
            h = F.dropout(h, self.drop_ratio, training=self.training)
            # Residual skip connection with scaling to stabilize deep propagation
            h = h + h_list[layer]
            h_list.append(h)

        # Jump Knowledge (JK) layer aggregation: Pool each layer's representations.
        # We include global sum pooling to capture molecular size/weight, which is critical for solubility.
        pooled_features = []
        for h_layer in h_list:
            h_mean = global_mean_pool(h_layer, batch)
            h_max = global_max_pool(h_layer, batch)
            h_sum = global_add_pool(h_layer, batch)
            pooled_features.append(torch.cat([h_mean, h_max, h_sum], dim=-1))
        
        # Concatenate final representation from initial, middle, and last layer groups to preserve diverse scales
        # and project back to the expected dimension.
        h_jk = torch.cat([pooled_features[0], pooled_features[self.num_layer // 2], pooled_features[-1]], dim=-1)
        
        h = self.jk_proj(h_jk)
        h = self.feat_lin(h)
        
        # Residual MLP head with SiLU activation
        h = h + F.silu(self.pred_head1(h))
        output = self.pred_head2(h)

        return output
# EVOLVE-BLOCK-END

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
