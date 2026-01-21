import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Take command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--meanrew", type=float, default=4.0)
parser.add_argument("--dataset", type=str, default="MeanReward_training_data.csv")

args = parser.parse_args()

# Setting amount of steps to Unit of time
TIME_UNIT = 60000
# Taking the input file and setting X and y as metrics
DIR = os.path.dirname(os.path.abspath(__file__))
INPUT=os.path.abspath(os.path.join(DIR, "..", "data", args.dataset))
df = pd.read_csv(INPUT)
X = df[['Mean Reward']]
y = df['Step']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Running GBDT with a quantile loss for different sectors to estimate earliest, mean and latest time step to achieve the reward
model_low = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=1000)
model_low.fit(X_train, y_train)

model_mid = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=1000)
model_mid.fit(X_train, y_train)

model_high = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=1000)
model_high.fit(X_train, y_train)

# Setting the prediction value
target_reward = args.meanrew
target = pd.DataFrame([[target_reward]], columns=['Mean Reward'])

# Predicting
step_early = model_low.predict(target)[0]
step_mid = model_mid.predict(target)[0]
step_late = model_high.predict(target)[0]

# Outputting the results
print(f"Prediction in time units rounded: {target_reward}")
print(f"Earliest possible: {int(round(step_early / TIME_UNIT))} TU")
print(f"Most likely: {int(round(step_mid / TIME_UNIT))} TU")
print(f"Latest possible: {int(round(step_late / TIME_UNIT))} TU")


# Plotting the learning curve
train_sizes, train_scores, test_scores = learning_curve(model_mid,X,y,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,train_sizes=np.linspace(0.1, 1.0, 5))

train_scores_mean = -np.mean(train_scores, axis=1) / TIME_UNIT
test_scores_mean = -np.mean(test_scores, axis=1) / TIME_UNIT

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Error")
plt.xlabel("Training Examples")
plt.ylabel("Error (Time Units)")
plt.title("Learning Curve: Training vs Validation Error (in Time Units)")
plt.legend(loc="best")
plt.grid()
plt.show()


# Method outputting the results
def get_metrics_tu(model, X_train, y_train, X_test, y_test, alpha):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_acc = np.mean(np.abs(y_test - test_pred) <= TIME_UNIT)
    return {"Alpha": alpha,"Train MAE in TU": mean_absolute_error(y_train, train_pred) / TIME_UNIT,"Test MAE in TU": mean_absolute_error(y_test, test_pred) / TIME_UNIT,"Test R^2": r2_score(y_test, test_pred),"Accuracy (+/- 1 TU)": f"{test_acc:.2%}"}

results = [
    get_metrics_tu(model_low, X_train, y_train, X_test, y_test, 0.1),
    get_metrics_tu(model_mid, X_train, y_train, X_test, y_test, 0.5),
    get_metrics_tu(model_high, X_train, y_train, X_test, y_test, 0.9)
]
print("Model Metrics in time units:")
print(pd.DataFrame(results).round(4).to_string(index=False))
