import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import joblib

DATA_PATH = "data/wp_model_data.pkl"
MODEL_OUTPUT_PATH = "./wp_model_xgb.joblib"

print("Loading data...")
df = pd.read_pickle(DATA_PATH)

features = [
    "yardline_100",
    "ydstogo",
    "down",
    "qtr",
    "quarter_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining"
]
target = "wp"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f}")

print(f"Saving model to {MODEL_OUTPUT_PATH}...")
joblib.dump(model, MODEL_OUTPUT_PATH)
print("Done.")

def predict_win_probability(game_state):
    """
    Predict win probability given a game state dictionary.
    The dictionary should contain:
    - yardline_100
    - ydstogo
    - down
    - qtr
    - quarter_seconds_remaining
    - score_differential
    - posteam_timeouts_remaining
    - defteam_timeouts_remaining
    """
    model = joblib.load(MODEL_OUTPUT_PATH)
    input_df = pd.DataFrame([{
        "yardline_100": game_state["yardline_100"],
        "ydstogo": game_state["ydstogo"],
        "down": game_state["down"],
        "qtr": game_state["qtr"],
        "quarter_seconds_remaining": game_state["quarter_seconds_remaining"],
        "score_differential": game_state["score_differential"],
        "posteam_timeouts_remaining": game_state["posteam_timeouts_remaining"],
        "defteam_timeouts_remaining": game_state["defteam_timeouts_remaining"]
    }])
    return model.predict(input_df)[0]
