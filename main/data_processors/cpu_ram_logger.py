import psutil
import time
import yaml
import argparse
import os
import csv

def find_process_by_name(name):
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and name.lower() in proc.info['name'].lower():
                return proc
        except psutil.NoSuchProcess:
            continue
    return None

def collect_metrics(proc):
    try:
        cpu = proc.cpu_percent(interval=None)
        mem_mb = proc.memory_info().rss / 1024 / 1024 # counted in MB
        return cpu, mem_mb
    except Exception:
        return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity-process-name", required=True)
    parser.add_argument("--python-process-name", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # load hyperparameters from auto-generated YAML files
    with open(args.yaml) as f:
        config = yaml.safe_load(f)
    # Here, I am extracting the most important features or categories of features
    # to use them for data collection once the backend training starts
    behavior_name = list(config["behaviors"].keys())[0]
    hyperparams = config["behaviors"][behavior_name]["hyperparameters"]
    network = config["behaviors"][behavior_name]["network_settings"]
    reward = config["behaviors"][behavior_name]["reward_signals"]["extrinsic"]
    timing = config["behaviors"][behavior_name]


    config_name = os.path.basename(args.yaml)
    total_ram_mb = psutil.virtual_memory().total/1024/1024
    cpu_cores = psutil.cpu_count(logical=True)


    # the code will wait until it finds Unity and ML-Agents PIDs before it starts collecting data
    print("Waiting for Unity and ML-Agents processes...")
    unity_proc = None
    ml_proc = None
    while not unity_proc or not ml_proc:
        if not unity_proc:
            unity_proc = find_process_by_name(args.unity_process_name)
        if not ml_proc:
            ml_proc = find_process_by_name(args.python_process_name)
        time.sleep(0.5)

    print(f"Found Unity PID: {unity_proc.pid}, ML-Agents PID: {ml_proc.pid}")

    cpu_unity_list, ram_unity_list = [], []
    cpu_python_list, ram_python_list = [], []

    timestamps = []

    while ml_proc.is_running():
        ts = time.time()

        cpu_u, ram_u = collect_metrics(unity_proc)
        cpu_p, ram_p = collect_metrics(ml_proc)

        timestamps.append(ts)

        cpu_unity_list.append(cpu_u)
        ram_unity_list.append(ram_u)
        cpu_python_list.append(cpu_p)
        ram_python_list.append(ram_p)

        time.sleep(args.interval)

    # Here, the code ensures that we will have at least one sample
    # It was created for testing purposes in order to see what happens if the execution is stopped or the machine crashes in the meantime
    # In these cases, the code will pretend to have collected 0-valued data
    if not cpu_unity_list:
        timestamps.append(time.time())
        cpu_unity_list.append(0.0)
        ram_unity_list.append(0.0)
        cpu_python_list.append(0.0)
        ram_python_list.append(0.0)

    duration_sec = timestamps[-1] - timestamps[0]

    row = [
        f"run_{int(time.time())}",
        config_name,
        total_ram_mb,
        cpu_cores,
        hyperparams.get("batch_size"),
        hyperparams.get("buffer_size"),
        hyperparams.get("learning_rate"),
        hyperparams.get("beta"),
        hyperparams.get("epsilon"),
        hyperparams.get("lambd"),
        hyperparams.get("num_epoch"),
        network.get("num_layers"),
        network.get("hidden_units"),
        reward.get("gamma"),
        reward.get("strength"),
        timing.get("time_horizon"),
        timing.get("summary_freq"),
        timing.get("keep_checkpoints"),
        timing.get("max_steps"),
        sum(cpu_unity_list)/len(cpu_unity_list),
        max(cpu_unity_list),
        sum(ram_unity_list)/len(ram_unity_list),
        max(ram_unity_list),
        sum(cpu_python_list)/len(cpu_python_list),
        max(cpu_python_list),
        sum(ram_python_list)/len(ram_python_list),
        max(ram_python_list),
        duration_sec
    ]

    # writes to a CSV file; same order followed as in the above row array
    file_exists = os.path.isfile(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                 "run_id", "config_name",
                 "total_ram_mb","cpu_cores",
                 "batch_size","buffer_size","learning_rate","beta","epsilon","lambd","num_epoch",
                 "num_layers","hidden_units","gamma","strength","time_horizon","summary_freq",
                 "keep_checkpoints","max_steps",
                 "cpu_avg_unity","cpu_peak_unity","ram_avg_unity","ram_peak_unity",
                 "cpu_avg_python","cpu_peak_python","ram_avg_python","ram_peak_python","duration_sec"
            ])
        writer.writerow(row)

    print(f"Metrics written to {args.output}")

if __name__ == "__main__":
    main()