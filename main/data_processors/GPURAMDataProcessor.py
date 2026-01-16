import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))#get the path of the current folder
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))#go back one step to reach main and go to the data folder
INPUT= os.path.join(DATA_DIR, "PushBlock-Dataset-20260115.csv")#file we get the data from(change if needed)
OUTPUT= os.path.join(DATA_DIR, "RamCpu_training_data.csv")#output file(change if needed)

df = pd.read_csv(INPUT)
#normalize cpu average for both unity and the python program
df["cpu_avg_python"] = (df["cpu_avg_python"] / df["cpu_cores"])
df["cpu_avg_unity"] = (df["cpu_avg_unity"] / df["cpu_cores"])
#normalize cpu peak for both unity and the python program
df["cpu_peak_python"] = (df["cpu_peak_python"] / df["cpu_cores"])
df["cpu_peak_unity"] = (df["cpu_peak_unity"] / df["cpu_cores"])

df.to_csv(OUTPUT, index=False)#save it in the data directory