import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, num_heads, alpha=0.2):
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

    def forward(self, input, adj, M, activation=None):
        # input (N, FIN)
        # W (FIN, NH * FOUT)

        # H (N, FIN) @ (FIN, NH * FOUT) -> (N, NH * FOUT) -> (N, NH, FOUT)
        h = torch.mm(input, self.W).view(-1, self.num_heads, self.out_features)

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
        h_prime = torch.matmul(attention, h.transpose(0, 1))

        return (
            activation(h_prime.view(-1, self.num_heads * self.out_features))
            if activation
            else h_prime.view(-1, self.num_heads * self.out_features)
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
