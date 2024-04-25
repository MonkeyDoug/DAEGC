import yaml
import subprocess

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
                    for dropout in [0.1, 0.2, 0.5]:
                        for alpha in [0.2]:
                            for pre_lr in [0.005]:
                                for lr in [0.0001]:
                                    for embedding_size in [16, 32, 8]:
                                        curr_hidden_sizes = [
                                            hidden_size + [embedding_size]
                                            for hidden_size in hidden_sizes
                                            if len(hidden_size) < MAX_GAT_LAYERS
                                        ]
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
                                                "dropout" : dropout
                                            }
                                            print(config)
                                            with open("config.yaml", "w") as f:
                                                yaml.dump(config, f)
                                            p = subprocess.run(
                                                ["python3", "pretrain.py"],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE,
                                                text=True,
                                            )
                                            if p != 0:
                                                print(p.stdout)
                                                print(p.stderr)
                                            if p.returncode == 0:
                                                p = subprocess.run(
                                                    ["python3", "daegc.py"],
                                                    stderr=subprocess.PIPE,
                                                    text=True,
                                                )
                                            if p != 0:
                                                print(p.stdout)
                                                print(p.stderr)
