import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from decision_simulator import simulate_decision

df = pd.read_pickle("data/full_data.pkl")

filtered = df[
    (df["down"] == 4) &
    (df["play_type"].isin(["punt", "field_goal", "run", "pass"]))
].copy()

def label_actual(play_type):
    if play_type in ["run", "pass"]:
        return "go"
    return play_type

filtered["actual_decision"] = filtered["play_type"].apply(label_actual)

def get_model_decision(row):
    game_state = {
        "yardline_100": row["yardline_100"],
        "ydstogo": row["ydstogo"],
        "qtr": row["qtr"],
        "quarter_seconds_remaining": row["quarter_seconds_remaining"],
        "score_differential": row["score_differential"],
        "posteam_timeouts_remaining": row["posteam_timeouts_remaining"],
        "defteam_timeouts_remaining": row["defteam_timeouts_remaining"],
        "down": 4
    }
    try:
        decisions = simulate_decision(game_state)
        sorted_decisions = sorted(decisions.items(), key=lambda x: x[1], reverse=True)
        if abs(sorted_decisions[0][1] - sorted_decisions[1][1]) < 0.015:
            return None  # Skip indecisive
        return sorted_decisions[0][0]
    except:
        return None

print("Labeling data using model...")
filtered["model_decision"] = filtered.apply(get_model_decision, axis=1)
labeled = filtered.dropna(subset=["model_decision"])

features = [
    "yardline_100", "ydstogo", "qtr", "quarter_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining"
]
X = labeled[features]
y = labeled["model_decision"]

print("Training classifier to replicate model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

X_train_display = X_train.rename(columns={
    "yardline_100": "Field Position",
    "ydstogo": "Yards to Go",
    "qtr": "Quarter",
    "quarter_seconds_remaining": "Time Remaining (Quarter)",
    "score_differential": "Score Differential",
    "posteam_timeouts_remaining": "Off Timeouts Left",
    "defteam_timeouts_remaining": "Def Timeouts Left"
})

class_names = [name.capitalize() for name in clf.classes_] 

print("Running SHAP analysis...")
explainer = shap.Explainer(clf, X_train_display)
shap_values = explainer(X_train_display)

shap.summary_plot(
    shap_values,
    X_train_display,
    class_names=class_names,
    plot_type="bar",
    show=False,
    color=plt.cm.Set1
)

plt.title("Feature Importance in 4th Down Decision Model")
plt.tight_layout()
plt.show()