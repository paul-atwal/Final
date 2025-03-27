import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np
import matplotlib.pyplot as plt
from go_model import predict_conversion_probability

# Create a grid of yardline_100 and ydstogo values
yardlines = np.arange(10, 100, 5)  # from 10 to 95
ydstogos = np.arange(1, 11)        # from 1 to 10

# Store results in a matrix
conversion_probs = np.zeros((len(ydstogos), len(yardlines)))

for i, ydstogo in enumerate(ydstogos):
    for j, yardline in enumerate(yardlines):
        prob = predict_conversion_probability(yardline, ydstogo)
        conversion_probs[i, j] = prob

# Plotting
plt.figure(figsize=(10, 6))
c = plt.imshow(conversion_probs, cmap="viridis", aspect="auto",
               extent=[yardlines[0], yardlines[-1], ydstogos[-1], ydstogos[0]])
plt.colorbar(c, label="Conversion Probability")
plt.title("4th Down Conversion Probability by Field Position and Yards to Go")
plt.xlabel("Yards from Opponent End Zone (yardline_100)")
plt.ylabel("Yards to Go")
plt.xticks(yardlines)
plt.yticks(ydstogos)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
