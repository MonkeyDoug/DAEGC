import yaml

config = {
    "dataset": "Citeseer",
    "epoch": 20,
    "max_epoch": 100,
    "pre_lr": 0.005,
    "lr": 0.0001,
    "n_clusters": 6,
    "update_interval": 1,
    "hidden_sizes": [256, 16],
    "embedding_size": 16,
    "weight_decay": "5E-3",
    "alpha": 0.2,
    "num_heads": 2,
    "num_gat_layers": 2,
}

MAX_GAT_LAYERS = 6

hidden_sizes = [128, 256, 512, 1024]

hidden_sizes = [
    [],
    [128],
    [256],
    [512],
    [128],
    [256],
    [512],
    [128] * 2,
    [256] * 2,
    [512] * 2,
    [1024] * 2,
    [128] * 3,
    [256] * 3,
    [512] * 3,
    [1024] * 3,
    [128] * 4,
    [256] * 4,
    [512] * 4,
    [1024] * 4,
    [128] * 5,
    [256] * 5,
    [512] * 5,
    [1024] * 5,
    [128] * 6,
    [256] * 6,
    [512] * 6,
    [1024] * 6,
    [128, 256, 256, 128],
    [128, 256, 512, 256, 128],
    [128, 256, 512, 512, 256, 128],
    [256, 512, 1024, 1024, 512, 256],
]

configs_finished = []

all_configs = []

for n_clusters in range(1, 11):
    for update_interval in range(1, 11):
        for max_epoch in range(100, 301, 100):
            for epoch in range(50, 101, 5):
                for num_heads in range(1, 6):
                    for alpha in [i / 100.0 for i in range(1, 51, 5)]:
                        for pre_lr in [i / 1000 for i in range(1, 11, 1)]:
                            for lr in [0.0001, 0.001, 0.0005]:
                                for embedding_size in [16, 32, 8, 64, 128]:
                                    curr_hidden_sizes = [
                                        hidden_size + [embedding_size]
                                        for hidden_size in hidden_sizes
                                        if len(hidden_size) < MAX_GAT_LAYERS
                                    ] + hidden_sizes
                                    for curr_hidden_size in curr_hidden_sizes:
                                        config = {
                                            "dataset": "Citeseer",
                                            "epoch": epoch,
                                            "max_epoch": max_epoch,
                                            "pre_lr": pre_lr,
                                            "lr": lr,
                                            "n_clusters": n_clusters,
                                            "update_interval": update_interval,
                                            "hidden_sizes": curr_hidden_size,
                                            "embedding_size": embedding_size,
                                            "weight_decay": "5E-3",
                                            "alpha": alpha,
                                            "num_heads": num_heads,
                                            "num_gat_layers": len(curr_hidden_size),
                                        }
                                        all_configs.append(config)

print(all_configs)
print(len(all_configs))

# with open("config.yaml", "r+") as f:
