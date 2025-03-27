import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decision_simulator import simulate_decision

# Load data
df = pd.read_pickle("data/full_data.pkl")

# Filter 4th down, in appropriate WP and yardline range
filtered = df[
    (df["down"] == 4) &
    (df["yardline_100"] < 50) &
    (df["ydstogo"] <= 3) &
    (df["wp"].between(0.2, 0.8)) &
    (df["play_type"].isin(["punt", "field_goal", "run", "pass"]))
].copy()

# Simplify actual decision
def label_actual(play):
    if play["play_type"] in ["run", "pass"]:
        return "go"
    elif play["play_type"] == "field_goal":
        return "field_goal"
    elif play["play_type"] == "punt":
        return "punt"
    return None

filtered["actual_decision"] = filtered.apply(label_actual, axis=1)

# Track results
season_results = {}

for season, group in filtered.groupby("season"):
    team_correct = {}
    team_total = {}

    for _, row in group.iterrows():
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
            model_wps = simulate_decision(game_state)
        except Exception:
            continue

        sorted_options = sorted(model_wps.items(), key=lambda x: x[1], reverse=True)
        best, second_best = sorted_options[0], sorted_options[1]

        # Skip if options are close
        if abs(best[1] - second_best[1]) < 0.015:
            continue

        model_choice = best[0]
        actual_choice = row["actual_decision"]

        if actual_choice is None:
            continue

        team = row["posteam"]
        if team not in team_correct:
            team_correct[team] = 0
            team_total[team] = 0

        if model_choice == actual_choice:
            team_correct[team] += 1
        team_total[team] += 1

    # Calculate accuracy
    team_accuracy = {
        team: team_correct[team] / team_total[team]
        for team in team_total if team_total[team] > 0
    }
    league_accuracy = sum(team_correct.values()) / sum(team_total.values()) if team_total else 0
    season_results[season] = team_accuracy

    print(f"Season {season}: League Accuracy = {league_accuracy:.3f}")

    # Plot
    # teams = list(team_accuracy.keys())
    # accuracies = [team_accuracy[t] for t in teams]
    # plt.figure(figsize=(10, 4))
    # sns.barplot(x=teams, y=accuracies)
    # plt.title(f"Team Decision Accuracy vs Bot ({season})")
    # plt.ylabel("Accuracy")
    # plt.xticks(rotation=90)
    # plt.ylim(0, 1)
    # plt.tight_layout()
    # plt.show()