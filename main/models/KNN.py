import argparse

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setting amount of steps to Unit of time
TIME_UNIT = 60000

# Take command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--meanrew", type=float, default=4.0)
parser.add_argument("--dataset", type=str, default="MeanReward_training_data.csv")

args = parser.parse_args()

# Taking the input file and setting X and y as metrics

DIR = os.path.dirname(os.path.abspath(__file__))#directory of current file
INPUT=os.path.abspath(os.path.join(DIR, "..", "data",args.dataset))#data found in the data directory
df = pd.read_csv(INPUT)
X = df[['Mean Reward']]
y = df['Step']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Running KNN
knn = KNeighborsRegressor(n_neighbors=100)
knn.fit(X_train, y_train)
target_reward = args.meanrew
target_df = pd.DataFrame([[target_reward]], columns=['Mean Reward'])

predicted_step = knn.predict(target_df)[0]
predicted_tu = int(round(predicted_step / TIME_UNIT))

# Displaying results
print(f"Prediction {target_reward}:")
print(f"Time Units {predicted_tu} TU")

# Method calculating the model's metrics
def get_knn_metrics(model, X_data, y_data, dataset_name):
    y_pred_raw = model.predict(X_data)

    y_true_tu = y_data / TIME_UNIT
    y_pred_tu = y_pred_raw / TIME_UNIT

    err = abs(y_true_tu - y_pred_tu)
    acc = np.mean(err <= 1.0)

    return {"Dataset": dataset_name,"MAE in TU": mean_absolute_error(y_data, y_pred_raw) / TIME_UNIT,"RMSE in TU": np.sqrt(mean_squared_error(y_data, y_pred_raw)) / TIME_UNIT,"R^2 Score": r2_score(y_data, y_pred_raw),"Accuracy (+/- 1 TU)": f"{acc:.2%}"}

results = [
    get_knn_metrics(knn, X_train, y_train, "Training"),
    get_knn_metrics(knn, X_test, y_test, "Validation")
]

# Displaying the model's performance
print("model performance")
print(pd.DataFrame(results).round(4).to_string(index=False))


train_sizes, train_scores, test_scores = learning_curve(knn,X,y,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,train_sizes=np.linspace(0.1, 1.0, 5))

train_scores_mean = -np.mean(train_scores, axis=1) / TIME_UNIT
test_scores_mean = -np.mean(test_scores, axis=1) / TIME_UNIT

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Error")
plt.xlabel("Training Examples")
plt.ylabel("Error (Time Units)")
plt.title("Learning Curve (KNN, k=100) - Error in Time Units")
plt.legend(loc="best")
plt.grid()
plt.show()
