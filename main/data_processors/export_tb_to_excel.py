import os, glob, math
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import sys

if len(sys.argv) > 1:
    ITERATION = sys.argv[1] #this will represent the threshold we attamepted to reach during the training
    TIMESTAMP =sys.argv[2] #this represents the differentiator between trainings to avoid conflicts
else:
    ITERATION = "default"

DIR =os.path.dirname(os.path.abspath(__file__))#path of data_processors directory
RESULT_DIR = os.path.join(DIR, "..","data", "rawMeanReward_data")#directory where we will put our collected trainings
os.makedirs(RESULT_DIR, exist_ok=True)#create directory if missing

REPO_ROOT = os.path.abspath(os.path.join(DIR, "..", ".."))#this way we can access the results folder from anywhere
LOGDIR = os.path.join( REPO_ROOT,"results",f"threshold_{ITERATION}_t_{TIMESTAMP}","PushBlock")#here we take the data from the results folder and process it
OUTFILE= os.path.join(RESULT_DIR,f"selected_training_data_threshold_{ITERATION}_t_{TIMESTAMP}.xlsx")#this will be the resulting file
STEP_BIN =5000

event_files = glob.glob(f"{LOGDIR}/**/events.out.tfevents.*", recursive=True)
if not event_files:
    print(f"No TensorBoard event files found under {LOGDIR}")
    raise SystemExit(1)

all_rows= []

#creates an EventAccumulator file the event file; after that it loads it into memory
#we look for the "cumulative reward" which will be the metric we measure(threshold)
#save the values we want to measure which will then be processed into an excel
for event_file in event_files:
    ea= event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    reward_tag =None
    for t in tags:#look for an exact match
        if t.strip().lower() =="environment/cumulative reward":
            reward_tag =t
            break
    if reward_tag is None:#looks for something similar
        candidates = [t for t in tags if "cumulative" in t.lower() and "reward" in t.lower()]
        if candidates:
            reward_tag =candidates[0]

    if reward_tag is None:#if there is nothing similar or exact match we skip the file
        print(f"No cumulative reward scalar in {event_file} Available: {tags}")
        continue

    events = ea.Scalars(reward_tag)
    for e in events:
        all_rows.append({"step": e.step,"value": e.value})

if not all_rows:
    print("Found no reward scalars to aggregate")
    raise SystemExit(1)

#sorts based on the steps(to make sure everything is ordered)
df = pd.DataFrame(all_rows).sort_values("step")
df["step_bin"] = (df["step"] // STEP_BIN) * STEP_BIN#assign steps to a bin

agg = df.groupby("step_bin").agg(
    **{
        "Mean Reward":("value", "mean"),#compute the mean from the value column using the bin

    }
).reset_index().rename(columns={"step_bin": "Step"})

#save the file to excel
agg.to_excel(
    OUTFILE,index=False, engine="openpyxl"
)

print(f"Wrote {len(agg)} rows to {os.path.abspath(OUTFILE)}")
