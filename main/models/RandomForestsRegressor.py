import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

SCRIPT_DIR= os.path.dirname(os.path.abspath(__file__))#current directory
INPUT =os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data","MeanReward_training_data.csv"))#reaching the data by going to the data directory

df = pd.read_csv(INPUT)

# prepare data
X = df[['Mean Reward']].values
y = df['Step'].values

# splitting data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nNum of training samples: {len(X_train)}")
print(f"Num of testing samples: {len(X_test)}")

# creating and training the regressor
rf_model =RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# actual predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# performance metrics for the model
training_r2_score = r2_score(y_train, y_pred_train)
testing_r2_score = r2_score(y_test, y_pred_test)

training_mean_abs_error = mean_absolute_error(y_train, y_pred_train)
testing_mean_abs_error = mean_absolute_error(y_test, y_pred_test)

training_root_mean_sq_error = np.sqrt(mean_squared_error(y_train, y_pred_train))
testing_root_mean_sq_error= np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Training accuracy {training_r2_score}")
print(f"Validation accuracy {testing_r2_score}")
print(f"Training mean abs error {training_mean_abs_error}")
print(f"Testing mean abs error {testing_mean_abs_error}")
print(f"Training root mean squared error {training_root_mean_sq_error}")
print(f"Testing root mean squared error {testing_root_mean_sq_error}")

# cross-validation implementation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()} ({cv_scores.std() * 2})")

# function to return prediction of steps for a target reward
def predict_steps_for_reward(target_reward):
    prediction = rf_model.predict([[target_reward]])[0]
    return int(prediction)

#example predictions
target_rewards = [2.0,3.0, 4.0, 5.0]
for reward in target_rewards:
    steps = predict_steps_for_reward(reward)
    print(f"To reach mean reward of {reward:.1f}: ~{steps:,} steps")
