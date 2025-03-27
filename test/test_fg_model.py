import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import matplotlib.pyplot as plt
import numpy as np
from fg_model import predict_fg_success_probability

# Generate yardline_100 values from 15 to 70
yardlines = np.arange(15, 71)
predicted_probs = [predict_fg_success_probability(yl) for yl in yardlines]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(yardlines, predicted_probs, label="FG Success Probability", color="blue")
plt.xlabel("Yards from Opponent End Zone (yardline_100)")
plt.ylabel("Predicted Field Goal Success Probability")
plt.title("Field Goal Success Probability by Distance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
