import yaml
import subprocess

hidden_sizes = [
    [128],
    [256],
    [384],
    [512],
    [256, 256],
    [512, 512],
    [1024, 1024],
]

for n_clusters in [6]:
    for update_interval in [1, 2, 5]:
        for max_epoch in [100, 200]:
            for epoch in [100]:
                for num_heads in [1, 2, 4]:
                    for dropout in [0.0, 0.1, 0.2, 0.5]:
                        for alpha in [0.2]:
                            for pre_lr in [0.005]:
                                for lr in [0.0001]:
                                    for embedding_size in [16, 32, 64]:
                                        curr_hidden_sizes = [
                                            hidden_size + [embedding_size]
                                            for hidden_size in hidden_sizes
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
                                                "dropout": dropout,
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
