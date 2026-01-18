import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

TIME_UNIT = 60000
DIR = os.path.dirname(os.path.abspath(__file__))
INPUT=os.path.abspath(os.path.join(DIR, "..", "data","Total_data_question_2.xslx"))
df = pd.read_csv(INPUT)
X = df[['Mean Reward']]
y = df['Step']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_estimators': [1000],'max_depth': [20],'min_samples_split': [75],'min_samples_leaf': [20]}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

target_reward = 4.0
prediction = best_rf.predict(pd.DataFrame([[target_reward]], columns=['Mean Reward']))[0]

print(f"Best Parameters Found: {grid_search.best_params_}")

def get_rf_metrics_tu(model, X_data, y_data, dataset_name):
    y_pred_raw = model.predict(X_data)

    y_true_tu = y_data / TIME_UNIT
    y_pred_tu = y_pred_raw / TIME_UNIT


    err = abs(y_true_tu - y_pred_tu)
    acc = np.mean(err <= 1.0)

    return {"Dataset": dataset_name,"MAE in TU": mean_absolute_error(y_data, y_pred_raw) / TIME_UNIT,"RMSE in TU": np.sqrt(mean_squared_error(y_data, y_pred_raw)) / TIME_UNIT,"R^2 Score": r2_score(y_data, y_pred_raw),"Accuracy (+/- 1 TU)": f"{acc:.2%}"}


results = [
    get_rf_metrics_tu(best_rf, X_train, y_train, "Training"),
    get_rf_metrics_tu(best_rf, X_test, y_test, "Validation (Test)")
]

print("Model Performance in time units:")
print(pd.DataFrame(results).round(4).to_string(index=False))


train_sizes, train_scores, test_scores = learning_curve(best_rf,X,y,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,train_sizes=np.linspace(0.1, 1.0, 5))

train_scores_mean = -np.mean(train_scores, axis=1) / TIME_UNIT
test_scores_mean = -np.mean(test_scores, axis=1) / TIME_UNIT

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Error")
plt.xlabel("Training Examples")
plt.ylabel("Error (Time Units)")
plt.title(f"Learning Curve (Depth={grid_search.best_params_['max_depth']}) - Error in Time Units")
plt.legend(loc="best")
plt.grid()
plt.show()
