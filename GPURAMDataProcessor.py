import pandas as pd

df = pd.read_csv("PushBlock-Final-Dataset-20260115.csv")
#normalize cpu average for both unity and the python program
df["cpu_avg_python"] = (df["cpu_avg_python"] / df["cpu_cores"])
df["cpu_avg_unity"] = (df["cpu_avg_unity"] / df["cpu_cores"])
#normalize cpu peak for both unity and the python program
df["cpu_peak_python"] = (df["cpu_peak_python"] / df["cpu_cores"])
df["cpu_peak_unity"] = (df["cpu_peak_unity"] / df["cpu_cores"])

df.to_csv("RamCpu_training_data.csv", index=False)