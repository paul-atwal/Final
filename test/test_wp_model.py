import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np
import matplotlib.pyplot as plt
from wp_model import predict_win_probability

# Fixed parameters for other variables
fixed_state = {
    "qtr": 3,
    "quarter_seconds_remaining": 900,
    "score_differential": 0,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "down": 1
}

# Vary yardline_100 and ydstogo
yardlines = np.arange(1, 101)
ydstogos = np.arange(1, 21)

wp_matrix = np.zeros((len(ydstogos), len(yardlines)))

for i, ydstogo in enumerate(ydstogos):
    for j, yline in enumerate(yardlines):
        state = fixed_state.copy()
        state["yardline_100"] = yline
        state["ydstogo"] = ydstogo
        wp_matrix[i, j] = predict_win_probability(state)

# Plot
plt.figure(figsize=(12, 6))
c = plt.imshow(wp_matrix, aspect='auto', cmap='coolwarm',
               extent=[yardlines[0], yardlines[-1], ydstogos[-1], ydstogos[0]])
plt.colorbar(c, label='Win Probability')
plt.title("Win Probability by Yardline and Yards to Go (Q3, Score Tied)")
plt.xlabel("Yards from Opponent End Zone (yardline_100)")
plt.ylabel("Yards to Go")
plt.tight_layout()
plt.show()
