import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import numpy as np
import os

# Logic to take parameters out of the command line and execute experiments
parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="cpu_avg_python")
parser.add_argument("--dataset", type=str, default="complete-datasetRAMCPU.csv")

args = parser.parse_args()

# Loads the training dataset
DIR = os.path.dirname(os.path.abspath(__file__))
INPUT= os.path.abspath(os.path.join(DIR, "..", "data",args.dataset))#we can also use the normalized data 
dataFile = pd.read_csv(INPUT)

# The target variable to be predicted
target = args.metric

# Remove ID columns and seperate features from targets
removeCols = ["run_id"]
x = dataFile.drop(columns=removeCols + [target])
y = dataFile[target]

# Convert  categorical features to category type for LightGBM
lightCols = []
if "config_name" in x.columns:
    x["config_name"] = x["config_name"].astype("category")
    lightCols.append("config_name")

# Split the data into training and validation sets (80% train, 20% validation)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Initialise the lightGBM model
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model and apply early stopping if validation scores dont improve for 200 rounds
model.fit(
    xTrain, yTrain,
    eval_set=[(xTest, yTest)],
    eval_metric="rmse",
    categorical_feature=lightCols if lightCols else "auto",
    callbacks=[lgb.early_stopping(stopping_rounds=200)]
)

# Generate the predictions
pred = model.predict(xTest)
rmse = np.sqrt(mean_squared_error(yTest, pred))
mae = mean_absolute_error(yTest, pred)
r2 = r2_score(yTest, pred)

# Print results
print("Target Column: " + target)
print("RMSE:",rmse, "\nMAE:", mae, "\nR^2:", r2)
