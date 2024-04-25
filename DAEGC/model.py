import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from layer import GATLayer


class GAT(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_sizes: List[int],
        embedding_size: int,
        num_gat_layers: int,
        num_heads: int,
        alpha: float,
        dropout: float,
    ) -> None:
        assert len(hidden_sizes) == num_gat_layers
        super(GAT, self).__init__()
        self.num_features = num_features
        self.hidden_sizes = hidden_sizes
        self.embedding_size = embedding_size
        self.num_gat_layers = num_gat_layers
        self.alpha = alpha

        self.lin_proj = nn.Linear(hidden_sizes[-1] * num_heads, embedding_size)

        layer_sizes = [num_features] + hidden_sizes

        self.gat_net = nn.ModuleList(
            [
                GATLayer(
                    in_features=(num_heads if i > 0 else 1) * layer_sizes[i],
                    out_features=layer_sizes[i + 1],
                    num_heads=num_heads,
                    concat=True if i < num_gat_layers - 1 else False,
                    activation=nn.ELU() if i < num_gat_layers - 1 else None,
                    add_skip_connection=False,
                    dropout=dropout,
                )
                for i in range(num_gat_layers)
            ]
        )

    def forward(self, x, adj, M):
        h = self.gat_net[0](x, adj, M)
        for i in range(1, self.num_gat_layers):
            h = self.gat_net[i](h, adj, M)

        # Comment out to average
        # (N, NH * H_LAST) -> (N, Embedding_Size)
        # h = self.lin_proj(h)

        # z = (N, Embedding_Size)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
