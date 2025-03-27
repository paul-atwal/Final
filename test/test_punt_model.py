import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import matplotlib.pyplot as plt
from punt_model import predict_opponent_yardline
import numpy as np

DATA_PATH = "data/punt_model_data.pkl"
 
df = pd.read_pickle(DATA_PATH)
punts = df.copy()

# Calculate actual opponent yardlines
punts["expected_opponent_yardline"] = (
    100 - punts["yardline_100"] + punts["kick_distance"] - punts["return_yards"]
)
punts = punts.dropna(subset=["yardline_100", "expected_opponent_yardline"])

# Actual means
actual_means = punts.groupby("yardline_100")["expected_opponent_yardline"].mean()

# Predicted values using punt_model
yardlines = np.arange(35, 101)
predicted = np.array([predict_opponent_yardline(y) for y in yardlines])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual_means.index, actual_means.values, label="Actual Mean", linewidth=2)
plt.plot(yardlines, predicted, label="Model Prediction", linestyle="--")
plt.xlabel("Yards from opponent's end zone (yardline_100)")
plt.ylabel("Expected Opponent Yardline After Punt")
plt.title("Punt Model vs. Actual Averages")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
