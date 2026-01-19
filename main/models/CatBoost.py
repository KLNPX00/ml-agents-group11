import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import os

# Logic to take parameters out of the command line and execute experiments
parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="cpu_avg_python")
parser.add_argument("--dataset", type=str, default="complete-datasetRAMCPU.csv")

args = parser.parse_args()

DIR = os.path.dirname(os.path.abspath(__file__))#current directory
INPUT= os.path.abspath(os.path.join(DIR, "..", "data",args.dataset))#we can also use the normalized data 
# the target column is what the model predicts for
# for the purpose of reproducability, I am thinking of moving this
# to a config file or a command-line argument

target_columns = [args.metric]
# target_columns = ["cpu_avg_python"] # backup in case I break it
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

score = r2_score(y_test, y_pred) # counts accuracy

print("Model: CatBoost")
print("Target:", target_columns)
print(f"R² score: {score:.4f}")