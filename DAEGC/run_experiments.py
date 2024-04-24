import yaml
import subprocess

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

MAX_GAT_LAYERS = 4

hidden_sizes = [
    # [],
    # [128],
    # [256],
    # [512],
    # [1024],
    # # [128] * 2,
    # # [256] * 2,
    # # [512] * 2,
    # # [1024] * 2,
    # [128] * 3,
    # [256] * 3,
    # [512] * 3,
    # [1024] * 3,
    # # [128] * 4,
    # # [256] * 4,
    # # [512] * 4,
    # # [1024] * 4,
    # [128] * 5,
    # [256] * 5,
    # [512] * 5,
    # [1024] * 5,
    [256],
    [256, 256],
    [256, 512],
    [1024, 1024],
    [256, 512, 256],
    # [128, 256, 256, 128],
    # [128, 256, 512, 256, 128],
    # [128, 256, 512, 512, 256, 128],
    # [256, 512, 1024, 1024, 512, 256],
]

configs_finished = []

all_configs = []

num_configs = 0

for n_clusters in [6]:
    for update_interval in range(1, 8, 2):
        for max_epoch in range(100, 201, 100):
            for epoch in [25, 50]:
                for num_heads in range(1, 5):
                    for alpha in [i / 100.0 for i in range(10, 31, 10)]:
                        for pre_lr in [0.005]:
                            for lr in [0.0001]:
                                for embedding_size in [16, 32, 8]:
                                    curr_hidden_sizes = [
                                        hidden_size + [embedding_size]
                                        for hidden_size in hidden_sizes
                                        if len(hidden_size) < MAX_GAT_LAYERS
                                    ]
                                    for curr_hidden_size in curr_hidden_sizes:
                                        print(config)
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
                                        with open("config.yaml", "w") as f:
                                            yaml.dump(config, f)
                                        p = subprocess.run(
                                            ["python3", "pretrain.py"],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                        )
                                        p = subprocess.run(
                                            ["python3", "daegc.py"],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                        )
