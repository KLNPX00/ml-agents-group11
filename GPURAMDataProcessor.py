import pandas as pd

df = pd.read_csv("PushBlock-Final-Dataset-20260115.csv")

df["cpu_avg_python"] = (df["cpu_avg_python"] / df["cpu_cores"])
df["cpu_avg_unity"] = (df["cpu_avg_unity"] / df["cpu_cores"])

df["cpu_peak_python"] = (df["cpu_peak_python"] / df["cpu_cores"])
df["cpu_peak_unity"] = (df["cpu_peak_unity"] / df["cpu_cores"])

df.to_csv("RamCpu_training_data.csv", index=False)