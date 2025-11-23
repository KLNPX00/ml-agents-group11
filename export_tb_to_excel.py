# export_selected_metrics_computed_to_excel.py
import os, glob, math
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import sys

# If a threshold is passed as a CLI argument
if len(sys.argv) > 1:
    ITERATION = sys.argv[1]  # example: "4.50"
else:
    ITERATION = "default"


ITERATION = 0
LOGDIR = f"results/threshold{ITERATION}Run/PushBlock"   # <-- adjust to your actual run folder
OUTFILE = f"selected_training_data_threshold{ITERATION}.xlsx"
STEP_BIN = 5000                     # window size for mean/std aggregation

# find all event files in the run
event_files = glob.glob(f"{LOGDIR}/**/events.out.tfevents.*", recursive=True)
if not event_files:
    print(f"❌ No TensorBoard event files found under {LOGDIR}")
    raise SystemExit(1)

all_rows = []

for event_file in event_files:
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    # pick the cumulative reward tag (exact or closest match)
    reward_tag = None
    for t in tags:
        if t.strip().lower() == "environment/cumulative reward":
            reward_tag = t
            break
    if reward_tag is None:
        # try fuzzy match
        candidates = [t for t in tags if "cumulative" in t.lower() and "reward" in t.lower()]
        if candidates:
            reward_tag = candidates[0]

    if reward_tag is None:
        print(f"⚠️ No cumulative reward scalar in {event_file}. Available: {tags}")
        continue

    events = ea.Scalars(reward_tag)
    # collect raw samples
    for e in events:
        all_rows.append({"step": e.step, "value": e.value, "wall_time": e.wall_time})

if not all_rows:
    print("❌ Found no reward scalars to aggregate.")
    raise SystemExit(1)

df = pd.DataFrame(all_rows).sort_values("step")

# time elapsed in seconds from start of run
t0 = df["wall_time"].min()
df["time_elapsed_s"] = df["wall_time"] - t0

# bin steps into windows (e.g., 0-4999, 5000-9999, ...)
df["step_bin"] = (df["step"] // STEP_BIN) * STEP_BIN

# aggregate mean/std per bin
agg = df.groupby("step_bin").agg(
    **{
        "Mean Reward": ("value", "mean"),
        "Std of Reward": ("value", "std"),
        # pick the earliest timestamp in the bin as representative elapsed time
        "Time Elapsed (s)": ("time_elapsed_s", "min"),
    }
).reset_index().rename(columns={"step_bin": "Step"})

# handle bins with a single sample (std = NaN) -> set to 0.0
agg["Std of Reward"] = agg["Std of Reward"].fillna(0.0)

# write to Excel (single sheet)
agg[["Step", "Time Elapsed (s)", "Mean Reward", "Std of Reward"]].to_excel(
    OUTFILE, index=False, engine="openpyxl"
)

print(f"✅ Wrote {len(agg)} rows to {os.path.abspath(OUTFILE)}")
print(f"   (window size = {STEP_BIN} steps; change STEP_BIN to smooth more/less)")
