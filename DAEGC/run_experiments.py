import sys
import yaml
import subprocess

hidden_sizes = [
    [128],
    [256],
    [512],
    [128, 128],
    [256, 256],
    [512, 512],
]

for add_skip_connection in [False, True]:
    for scheduler in [True, False]:
        for weight_decay in ["5E-4", "5E-3"]:
            for update_interval in [1, 2, 5]:
                for max_epoch in [100, 200]:
                    for epoch in [100]:
                        for k in [10, 5, 15]:
                            for dropout in [0.0, 0.1, 0.2, 0.5]:
                                for alpha in [0.2]:
                                    for num_heads in [1, 2, 4]:
                                        for embedding_size in [32, 16, 64]:
                                            curr_hidden_sizes = [
                                                hidden_size + [embedding_size]
                                                for hidden_size in hidden_sizes
                                            ]
                                            for curr_hidden_size in curr_hidden_sizes:
                                                config = {
                                                    "dataset": sys.argv[1],
                                                    "epoch": epoch,
                                                    "max_epoch": max_epoch,
                                                    "update_interval": update_interval,
                                                    "hidden_sizes": curr_hidden_size,
                                                    "embedding_size": embedding_size,
                                                    "weight_decay": "5E-4",
                                                    "alpha": alpha,
                                                    "num_heads": num_heads,
                                                    "num_gat_layers": len(
                                                        curr_hidden_size
                                                    ),
                                                    "dropout": dropout,
                                                    "k": k,
                                                    "scheduler": scheduler,
                                                    "add_skip_connection": add_skip_connection,
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
                                                if p.returncode != 0:
                                                    print(p.stdout)
                                                    print(p.stderr)
                                                if p.returncode == 0:
                                                    p = subprocess.run(
                                                        ["python3", "daegc.py"],
                                                        stderr=subprocess.PIPE,
                                                        text=True,
                                                    )
                                                if p.returncode != 0:
                                                    print(p.stdout)
                                                    print(p.stderr)
