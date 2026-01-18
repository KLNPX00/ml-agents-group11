import os
import yaml
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))#get the current directory
OUTPUT_DIR=os.path.abspath(os.path.join(DIR, "..", "..","configs"))# this directory has to be addressed as it is different from Linux; the configs folder is not in the same subfolder, but many folders out
num_samples = 60000
samples_per_round = 1000
"""
In the Linux machine we ran the code, the amount of available cpu cores was 6.
But in different Linux environments, it will be different.
Windows probably takes the default 1.
"""
cpu_cores = 6
np.random.seed(42)

param_ranges = {
    "num_layers": (1, 4),
    "hidden_units": (64, 512),
    "batch_size": (64, 1024),
    "buffer_size": (1024, 16384),
    "num_epoch": (1, 10),
    "num_envs": (1, cpu_cores),
    "time_horizon": (16, 128),
    "summary_freq": (500, 5000)
}

categorical_vars = {
    "normalize": [True, False],
    "vis_encode_type": ["simple"]
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LHS Latin Hypercube Sampling (I  have to add a brief explanation here about it. What is it? How it works? Why is it needed?)
def lhs_numpy(samples, dimensions):
    cut = np.linspace(0, 1, samples + 1)
    result = np.zeros((samples, dimensions))
    for j in range(dimensions):
        u = np.random.rand(samples)
        result[:, j] = u * (cut[1:samples+1] - cut[:samples]) + cut[:samples]
        np.random.shuffle(result[:, j])
    return result

num_params = len(param_ranges)
lhs_samples = lhs_numpy(num_samples, num_params)
param_names = list(param_ranges.keys())

# config generator function
for i in range(num_samples):
    sample = {}
    for j, pname in enumerate(param_names):
        low, high = param_ranges[pname]
        val = lhs_samples[i, j] * (high - low) + low
        val = int(round(val))
        sample[pname] = val

    # fix constraints
    if sample["batch_size"] > sample["buffer_size"]:
        sample["batch_size"] = sample["buffer_size"]
    if sample["num_envs"] > cpu_cores:
        sample["num_envs"] = cpu_cores

    # add categorical variables
    for cat_var, options in categorical_vars.items():
        val = np.random.choice(options)
        # convert to native Python type
        if isinstance(val, np.generic):
            val = val.item()
        sample[cat_var] = val

    file_num = i + 1

    round_number = (file_num - 1) // samples_per_round + 1
    ROUND_DIR = os.path.join(OUTPUT_DIR, f"round{round_number}")#we choose out of round 1 or 2
    os.makedirs(ROUND_DIR, exist_ok=True)

    yaml_path = os.path.join(ROUND_DIR, f"sample_{file_num}.yaml")


    # YAML structure as provided by Unity's ML-Agents
    # Variables that are not included here take default values
    yaml_dict = {
        "behaviors": {
            "PushBlock": {
                "trainer_type": "ppo",
                "hyperparameters": {
                    "batch_size": sample["batch_size"],
                    "buffer_size": sample["buffer_size"],
                    "learning_rate": 0.0003,
                    "beta": 0.01,
                    "epsilon": 0.3,
                    "lambd": 0.99,
                    "num_epoch": sample["num_epoch"],
                    "learning_rate_schedule": "linear"
                },
                "network_settings": {
                    "normalize": sample["normalize"],
                    "hidden_units": sample["hidden_units"],
                    "num_layers": sample["num_layers"],
                    "vis_encode_type": sample["vis_encode_type"]
                },
                "reward_signals": {
                    "extrinsic": {
                        "gamma": 0.999,
                        "strength": 0.5
                    }
                },
                "keep_checkpoints": 5,
                "max_steps": 50000,
                "time_horizon": sample["time_horizon"],
                "summary_freq": sample["summary_freq"]
            }
        }
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

print(f"Generated {num_samples} YAML files in {OUTPUT_DIR}")