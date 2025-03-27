import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from data_loader import load_nflfastR_data

DATA_PATH = "data/fg_model_data.pkl"
MODEL_OUTPUT_PATH = "./fg_model_logreg.joblib"

def train_fg_model():
    print("Loading data...")
    df = pd.read_pickle(DATA_PATH)

    fg_attempts = df.copy()

    # Label success = 1, miss = 0
    fg_attempts["fg_success"] = fg_attempts["field_goal_result"].apply(
        lambda x: 1 if x == "made" else 0
    )

    X = fg_attempts[["yardline_100"]]
    y = fg_attempts["fg_success"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training logistic regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Test AUC: {auc:.4f}")
 
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Saved model to {MODEL_OUTPUT_PATH}")

def predict_fg_success_probability(yardline_100):
    model = joblib.load(MODEL_OUTPUT_PATH)
    input_df = pd.DataFrame([{"yardline_100": yardline_100}])
    return model.predict_proba(input_df)[0][1]

if __name__ == "__main__":
    train_fg_model()
