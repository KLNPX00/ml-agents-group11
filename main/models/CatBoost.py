import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

csv_path = "complete-dataset.csv"
# the target column is what the model predicts for
# for the purpose of reproducability, I am thinking of moving this
# to a config file or a command-line argument
target_columns = ["cpu_avg_python"]
test_size = 0.2
random_state = 42

df = pd.read_csv(csv_path)

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