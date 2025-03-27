import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from decision_simulator import simulate_decision

def print_decision_outputs(situations):
    for i, situation in enumerate(situations, 1):
        print(f"--- Situation {i} ---")
        print(f"Description: {situation['description']}")
        game_state = {
            "yardline_100": situation["yardline_100"],
            "ydstogo": situation["ydstogo"],
            "qtr": situation["qtr"],
            "quarter_seconds_remaining": situation["quarter_seconds_remaining"],
            "score_differential": situation["score_differential"],
            "posteam_timeouts_remaining": 3,
            "defteam_timeouts_remaining": 3,
            "down": 4
        }
        results = simulate_decision(game_state)
        for decision, wp in results.items():
            print(f"{decision.capitalize()}: {wp:.4f}")
        print()

def main():
    sample_situations = [
        {
            "description": "4th & 2 from own 10, Q1, 10 min left, tied game",
            "yardline_100": 90,
            "ydstogo": 2,
            "qtr": 1,
            "quarter_seconds_remaining": 600,
            "score_differential": 0
        },
        {
            "description": "4th & 5 from own 12, Q1, 6 min left, trailing by 3",
            "yardline_100": 88,
            "ydstogo": 5,
            "qtr": 1,
            "quarter_seconds_remaining": 360,
            "score_differential": -3
        },
        {
            "description": "4th & 1 from own 13, Q1, 12 min left, tied game",
            "yardline_100": 87,
            "ydstogo": 1,
            "qtr": 1,
            "quarter_seconds_remaining": 720,
            "score_differential": 0
        },
        {
            "description": "4th & 10 from own 5, Q4, 2 min left, down by 7",
            "yardline_100": 95,
            "ydstogo": 10,
            "qtr": 4,
            "quarter_seconds_remaining": 120,
            "score_differential": -7
        },
        {
            "description": "4th & Goal from own 1, Q2, 2 min left, down by 4",
            "yardline_100": 99,
            "ydstogo": 1,
            "qtr": 2,
            "quarter_seconds_remaining": 120,
            "score_differential": -4
        }
    ]

    print_decision_outputs(sample_situations)

if __name__ == "__main__":
    main()
