import os
import argparse
import numpy as np

import wandb
import yaml


from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from torch_geometric.datasets import Planetoid

import utils
from model import GAT
from evaluation import eva


class DAEGC(nn.Module):
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
        super(DAEGC, self).__init__()
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
            add_skip_connection
        )
        self.gat.load_state_dict(torch.load(pretrain_path, map_location="cpu"))

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


def trainer(dataset, config):
    model = DAEGC(
        num_features=config["input_dim"],
        hidden_size=config["hidden_sizes"],
        embedding_size=config["embedding_size"],
        alpha=config["alpha"],
        num_clusters=config["n_clusters"],
        num_heads=config["num_heads"],
        num_gat_layers=config["num_gat_layers"],
        pretrain_path=config["pretrain_path"],
        dropout=config["dropout"],
        add_skip_connection=config["add_skip_connection"],
    ).to(device)
    print(model)

    # wandb configurations
    run_name = f'DAEGC Model INPUT_DIM: {config["input_dim"]} HIDDEN_DIM: {config["hidden_sizes"]} EMBEDDING_DIM: {config["embedding_size"]} ALPHA: {config["alpha"]} NUM_GAT LAYERS: {config["num_gat_layers"]} NUM_HEADS: {config["num_heads"]}'

    wandb.login(key="57127ebf2a35438d2137d5bed09ca5e4c5191ab9", relogin=True)

    run = wandb.init(
        name=run_name,
        reinit=True,
        project=f"10701-Project-{config['dataset']}",
        config=config,
        tags=["DAEGC"],
    )

    model_arch = str(model)
    model_arch_dir = os.path.join("model_archs", "daegc")
    if not os.path.exists(model_arch_dir):
        os.makedirs(model_arch_dir)
    with open("model_archs/daegc/daegc_model_arch.txt", "w") as f:
        f.write(model_arch)
    wandb.save("model_archs/daegc/daegc_model_arch.txt")

    best_overall_acc = 0.0
    best_overall_acc_path = os.path.join("best_overall_daegc_acc.txt")

    if os.path.exists(best_overall_acc_path):
        with open(best_overall_acc_path, "r") as f:
            best_overall_acc = float(f.read())

    optimizer = Adam(
        model.parameters(), lr=config["lr"], weight_decay=float(config["weight_decay"])
    )
    if config["scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=config["lr"] / 10
        )

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    with torch.no_grad():
        _, z = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=config["n_clusters"], n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, "pretrain")

    best_acc = 0.0

    for epoch in range(config["max_epoch"]):
        curr_lr = float(optimizer.param_groups[0]["lr"])

        if epoch % config["update_interval"] == 0:
            model.eval()
            # update_interval
            A_pred, z, Q = model(data, adj, M)

            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(y, q, epoch)

            wandb.log({"accuracy": acc, "nmi": nmi, "ari": ari, "f1": f1}, step=epoch)
            if acc > best_acc:
                best_acc = acc
                wandb.log({
                    'best_acc': best_acc
                }, step=epoch)

            if acc > best_overall_acc:
                best_overall_acc = acc
                torch.save(model.state_dict(), f"daegc_best_model.pkl")

        model.train()
        A_pred, z, q = model(data, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = config["k"] * kl_loss + re_loss

        wandb.log({"loss": loss, "learning_rate": curr_lr}, step=epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if config["scheduler"]:
            scheduler.step(epoch)

    with open(best_overall_acc_path, "w") as f:
        f.write(str(best_overall_acc))


if __name__ == "__main__":
    # uncomment below to download PubMed"
    # dataset = Planetoid(root='dataset', name="PubMed")

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = utils.get_dataset(config["dataset"])
    dataset = datasets[0]

    if config["dataset"] == "Citeseer":
        config["lr"] = 0.0001
        config["n_clusters"] = 6
    elif config["dataset"] == "Cora":
        config["lr"] = 0.0001
        config["n_clusters"] = 7
    elif config["dataset"] == "Pubmed":
        config["lr"] = 0.001
        config["n_clusters"] = 3

    e, dataset_name = config["epoch"], config["dataset"]
    config["pretrain_path"] = f"./pretrain/predaegc_best_model.pkl"
    config["input_dim"] = dataset.num_features

    trainer(dataset, config)
