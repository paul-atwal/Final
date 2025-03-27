import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

DATA_PATH = "data/go_model_data.pkl"
MODEL_OUTPUT_PATH = "./go_model_logreg.joblib"

def train_go_model():
    print("Loading data...")
    df = pd.read_pickle(DATA_PATH)

    go_plays = df.copy()

    # Label conversion success
    go_plays["converted"] = (go_plays["yards_gained"] >= go_plays["ydstogo"]).astype(int)

    X = go_plays[["yardline_100", "ydstogo"]]
    y = go_plays["converted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training logistic regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Test AUC: {auc:.4f}")

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Saved model to {MODEL_OUTPUT_PATH}")

def predict_conversion_probability(yardline_100, ydstogo):
    model = joblib.load(MODEL_OUTPUT_PATH)
    input_df = pd.DataFrame([{"yardline_100": yardline_100, "ydstogo": ydstogo}])
    return model.predict_proba(input_df)[0][1]

if __name__ == "__main__":
    train_go_model()
