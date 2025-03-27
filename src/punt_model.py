import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os

DATA_PATH = "data/punt_model_data.pkl"
MODEL_OUTPUT_PATH = "./punt_model_linear.joblib"

def train_punt_model():
    print("Loading data...")
    df = pd.read_pickle(DATA_PATH)

    punts = df.copy()
    punts = punts[punts["yardline_100"] >= 35]  # Only include punts from realistic field positions

    # Calculate expected opponent yardline
    punts["expected_opponent_yardline"] = (
        100 - punts["yardline_100"] + punts["kick_distance"]
    )

    # Drop rows with missing or invalid values
    punts = punts.dropna(subset=["yardline_100", "expected_opponent_yardline"])

    X = punts[["yardline_100"]]
    y = punts["expected_opponent_yardline"]

    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)

    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("Done.")

def predict_opponent_yardline(yardline_100):
    model = joblib.load(MODEL_OUTPUT_PATH)
    input_df = pd.DataFrame([{"yardline_100": yardline_100}])
    return model.predict(input_df)[0]

if __name__ == "__main__":
    train_punt_model()
