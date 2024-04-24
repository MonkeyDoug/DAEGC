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

import utils
from model import GAT

from evaluation import eva


def pretrain(dataset, config):
    model = GAT(
        num_features=config['input_dim'],
        hidden_sizes=config['hidden_sizes'],
        embedding_size=config['embedding_size'],
        alpha=config['alpha'],
        num_heads=config['num_heads'],
        num_gat_layers=config['num_gat_layers'],
    ).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=config['pre_lr'], weight_decay=float(config['weight_decay']))

    run_name = f'GAT PRETRAIN Model INPUT_DIM: {config["input_dim"]} HIDDEN_DIM: {config["hidden_sizes"]} EMBEDDING_DIM: {config["embedding_size"]} ALPHA: {config["alpha"]} NUM_GAT LAYERS: {config["num_gat_layers"]} NUM_HEADS: {config["num_heads"]}'

    wandb.login(key="57127ebf2a35438d2137d5bed09ca5e4c5191ab9", relogin=True)

    run = wandb.init(
        name=run_name,
        reinit=True,
        project="10701-Project",
        config=config
    )

    model_arch = str(model)
    model_arch_dir = os.path.join("model_archs", "gat_pretrain")
    if not os.path.exists(model_arch_dir):
        os.makedirs(model_arch_dir)
    with open("model_archs/gat_pretrain/gat_model_arch.txt", "w") as f:
        f.write(model_arch)
    wandb.save("model_archs/gat_pretrain/gat_model_arch.txt")

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    for epoch in range(config['max_epoch']):
        curr_lr = float(optimizer.param_groups[0]["lr"])

        model.train()
        A_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=config['n_clusters'], n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)

            wandb.log({"accuracy": acc,
                       "nmi": nmi,
                       "ari": ari,
                       "f1": f1,
                       "loss": loss,
                       "learning_rate": curr_lr})

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), f"./pretrain/predaegc_{config['dataset']}_{epoch}.pkl"
            )


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    datasets = utils.get_dataset(config['dataset'])
    dataset = datasets[0]

    if config['dataset'] == 'Citeseer':
        config['pre_lr'] = 0.005
        config['n_clusters'] = 6
    elif config['dataset'] == 'Cora':
        config['pre_lr'] = 0.005
        config['n_clusters'] = 7
    elif config['dataset'] == "Pubmed":
        config['pre_lr'] = 0.001
        config['n_clusters'] = 3
    
    config['input_dim'] = dataset.num_features
    
    print(config)
    pretrain(dataset, config)
