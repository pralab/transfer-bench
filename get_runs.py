"""Example script to get data from W&B."""

from benchmark.wandb_reader import WandbReader

reader = WandbReader("mlsec", "transfer-bench")
configs, states = reader.get_configs(), reader.get_runs_states()

for c, s in zip(configs, states, strict=True):
    print(c, s)
