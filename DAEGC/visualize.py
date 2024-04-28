import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


import utils
from model import GAT
from evaluation import eva


class MADGC(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size,
        embedding_size,
        alpha,
        num_clusters,
        num_gat_layers,
        num_heads,
        pretrain_path,
        dropout,
        add_skip_connection,
        v=1,
    ):
        super(MADGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(
            num_features,
            hidden_size,
            embedding_size,
            num_gat_layers,
            num_heads,
            alpha,
            dropout,
            add_skip_connection,
        )

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v
        )
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def t_sne(embeds, labels, name, sample_num=2000, show_fig=True):
    """
    visualize embedding by t-SNE algorithm
    :param embeds: embedding of the data
    :param labels: labels
    :param sample_num: the num of samples
    :param show_fig: if show the figure
    :return fig: figure
    """

    # sampling
    sample_index = np.random.randint(0, embeds.shape[0], sample_num)
    sample_embeds = embeds[sample_index]
    sample_labels = labels[sample_index]
    # sample_embeds = embeds
    # sample_labels = labels

    # t-SNE
    ts = TSNE(n_components=2, init="pca", random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds[:, :])

    fig = plt.figure()

    for i in range(ts_embeds.shape[0]):
        plt.scatter(
            ts_embeds[i, 0], ts_embeds[i, 1], s=45, c=plt.cm.Set1(sample_labels[i] % 7)
        )

    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    if show_fig:
        plt.show()
    plt.savefig(name)
    return fig


def trainer(name, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "Cora":
        model = MADGC(
            num_features=1433,
            hidden_size=[128, 128, 64],
            embedding_size=64,
            alpha=0.2,
            num_clusters=7,
            num_heads=1,
            num_gat_layers=3,
            pretrain_path="",
            dropout=0.1,
            add_skip_connection=False,
        ).to(device)
    if name == "Pubmed":
        # Too big to fit on GPU
        device = torch.device("cpu")
        model = MADGC(
            num_features=500,
            hidden_size=[2048, 128],
            embedding_size=128,
            alpha=0.2,
            num_clusters=3,
            num_heads=2,
            num_gat_layers=2,
            pretrain_path="",
            dropout=0,
            add_skip_connection=False,
        ).to(device)
    if name == "Citeseer":
        model = MADGC(
            num_features=3703,
            hidden_size=[128, 64],
            embedding_size=64,
            alpha=0.2,
            num_clusters=6,
            num_heads=4,
            num_gat_layers=2,
            pretrain_path="",
            dropout=0,
            add_skip_connection=False,
        ).to(device)
    print(model)

    model.load_state_dict(torch.load(f"{name.lower()}_best_model.pkl"))

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    model.eval()
    # update_interval
    A_pred, z, Q = model(data, adj, M)

    q = Q.detach().data.cpu().numpy().argmax(1)  # Q
    acc, nmi, ari, f1 = eva(y, q, 0)

    print(acc, nmi, ari, f1)

    t_sne(z.detach().numpy(), y, f"{name}.png", show_fig = False)


if __name__ == "__main__":
    datasets = utils.get_dataset(sys.argv[1])
    dataset = datasets[0]
    trainer(sys.argv[1], dataset, {})
