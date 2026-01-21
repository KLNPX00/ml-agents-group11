import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, root_mean_squared_error)
from catboost import CatBoostRegressor
import os

"""
Logic to get arguments from the command line to run different experiments.
It aims to facilitate the code running process for new users and avoid the
need to update parameters inside the code, but provide a more organized
way through a documented command line. Also, the two parameters that we 
used for our experiments in the CatBoost were the metric (for the processing 
metric we predict for) and dataset (for the dataset the prediction will 
happen based on).
"""
parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="cpu_avg_python")
parser.add_argument("--dataset", type=str, default="complete-datasetRAMCPU.csv")

args = parser.parse_args()

DIR = os.path.dirname(os.path.abspath(__file__))#current directory
INPUT= os.path.abspath(os.path.join(DIR, "..", "data",args.dataset))#we can also use the normalized data 

target_columns = [args.metric]
test_size = 0.2
random_state = 42

df = pd.read_csv(INPUT)

# keeps only numeric columns
df = df.select_dtypes(include=["int64", "float64"])

# drops rows that are empty
df = df.dropna()

# split features (for training)/targets (for predictions)
X = df.drop(columns=target_columns)
y = df[target_columns]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    random_seed=random_state,
    verbose=False
)

model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print("Model: CatBoost")
print("Target:", target_columns)
print(f"R^2 score: {score:.4f}")

print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))