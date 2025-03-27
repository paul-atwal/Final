import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from decision_simulator import simulate_decision

# Set fixed game state values
base_game_state = {
    "qtr": 2,
    "quarter_seconds_remaining": 600,  # 10 minutes
    "score_differential": 0,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3
}

# Define range of distances to end zone (yardline_100) and yards to go
yardline_vals = np.arange(100, 0, -5)
ydstogo_vals = np.arange(1, 11)

# Prepare grid to store recommendations
decisions_grid = []

for ydstogo in ydstogo_vals:
    row = []
    for yardline_100 in yardline_vals:
        game_state = base_game_state.copy()
        game_state["yardline_100"] = yardline_100
        game_state["ydstogo"] = ydstogo

        result = simulate_decision(game_state)
        best_decision = max(result, key=result.get)
        row.append(best_decision)
    decisions_grid.append(row)

# Convert to NumPy array for plotting
decision_array = np.array(decisions_grid)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    [[{"go": 0, "field_goal": 1, "punt": 2}[d] for d in row] for row in decision_array],
    cmap=["#d95f02", "#1b9e77", "#7570b3"],
    cbar=False,
    xticklabels=yardline_vals,
    yticklabels=ydstogo_vals
)

plt.title("4th Down Decision Model", fontsize=16, weight="bold")
plt.xlabel("Distance to Opponent End Zone (yardline_100)")
plt.ylabel("Yards to Go")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Add text labels for decisions
for i in range(len(ydstogo_vals)):
    for j in range(len(yardline_vals)):
        decision = decision_array[i][j]
        plt.text(j + 0.5, i + 0.5, decision.replace("_", " ").title(),
                 ha="center", va="center", fontsize=6, color="black")

plt.tight_layout()
plt.show()
