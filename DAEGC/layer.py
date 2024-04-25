# TODO
# Fix training accuracy
# How does eval work

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    head_dim = 1

    def __init__(
        self,
        in_features,
        out_features,
        num_heads,
        concat,
        activation,
        add_skip_connection,
        alpha=0.2,
        dropout=0.6,
    ):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout)

        self.add_skip_connection = add_skip_connection

        self.skip_proj = None

        if self.add_skip_connection:
            self.skip_proj = nn.Linear(
                in_features, num_heads * out_features, bias=False
            )

        self.concat = concat

        self.activation = activation

        self.bias = False

        self.lin_proj = nn.Linear(num_heads * out_features, out_features)

    def skip_concat_bias(self, in_nodes_features, out_nodes_features):
        if self.add_skip_connection:  # add skip or residual connection
            if (
                out_nodes_features.shape[-1] == in_nodes_features.shape[-1]
            ):  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(
                    -1, self.num_heads, self.out_features
                )

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(
                -1, self.num_heads * self.out_features
            )
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return (
            out_nodes_features
            if self.activation is None
            else self.activation(out_nodes_features)
        )

    def forward(self, input, adj, M):
        # input (N, FIN)
        # W (FIN, NH * FOUT)

        input = self.dropout(input)

        # H (N, FIN) @ (FIN, NH * FOUT) -> (N, NH * FOUT) -> (N, NH, FOUT)
        h = torch.mm(input, self.W).view(-1, self.num_heads, self.out_features)

        h = self.dropout(h)

        # H (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        attn_for_self = torch.sum((h * self.a_self), dim=-1, keepdim=True)
        attn_for_neighs = torch.sum((h * self.a_neighs), dim=-1, keepdim=True)

        attn_dense = attn_for_self.transpose(0, 1) + attn_for_neighs.permute(1, 2, 0)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (NH,N,N)

        zero_vec = -9e15 * torch.ones(
            (self.num_heads, adj.shape[0], adj.shape[1]), device=adj.device
        )
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=-1)

        # Attention (NH, N, N)
        # H (N, NH, FOUT) -> (NH, N, FOUT)

        # H_PRIME (NH, N, N) @ (NH, N, FOUT) -> (NH, N, FOUT)
        h_prime = torch.bmm(attention, h.transpose(0, 1))

        h_prime = h_prime.permute(1, 0, 2)

        if not h_prime.is_contiguous():
            h_prime = h_prime.contiguous()

        h_prime = self.skip_concat_bias(input, h_prime)

        return h_prime
