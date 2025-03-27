import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from decision_simulator import simulate_decision

st.title("NFL 4th Down Decision Model")

st.sidebar.header("Game Situation")

yardline = st.sidebar.slider("Yardline (distance to end zone)", 1, 99, 60)
ydstogo = st.sidebar.slider("Yards to Go", 1, 20, 4)
qtr = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])
quarter_seconds_remaining = st.sidebar.slider("Seconds left in Quarter", 0, 900, 300)
score_diff = st.sidebar.slider("Score Differential (your team)", -30, 30, 0)
posteam_timeouts = st.sidebar.selectbox("Your Timeouts Left", [0, 1, 2, 3], index=3)
defteam_timeouts = st.sidebar.selectbox("Opponent Timeouts Left", [0, 1, 2, 3], index=3)

if st.button("Run Simulation"):
    game_state = {
        "yardline_100": yardline,
        "ydstogo": ydstogo,
        "qtr": qtr,
        "quarter_seconds_remaining": quarter_seconds_remaining,
        "score_differential": score_diff,
        "posteam_timeouts_remaining": posteam_timeouts,
        "defteam_timeouts_remaining": defteam_timeouts,
        "down": 4
    }

    results = simulate_decision(game_state)

    # Filter punt when too close
    if yardline <= 35:
        results.pop("punt", None)

    # Filter field goal when too far
    if yardline > 60:
        results.pop("field_goal", None)

    st.subheader("Win Probability Estimates:")
    for option, wp in results.items():
        label = option.replace("_", " ").title()
        st.write(f"**{label}**: {wp * 100:.1f}%")

    best_choice = max(results, key=results.get)

    sorted_choices = sorted(results.items(), key=lambda x: x[1], reverse=True)
    margin = sorted_choices[0][1] - sorted_choices[1][1] if len(sorted_choices) > 1 else 0

    if margin >= 0.015:
        color = "green"
        confidence_text = "High confidence recommendation"
    else:
        color = "orange"
        confidence_text = "Options are close – lower confidence recommendation"

    st.markdown(f"<h3 style='color:{color};'>Recommended: {best_choice.upper()}</h3>", unsafe_allow_html=True)
    st.markdown(f"<i style='color:{color};'>{confidence_text}</i>", unsafe_allow_html=True)

    if best_choice == "go" and yardline > 85:
        st.markdown("Note: Model predictions may be less reliable when backed up near your own end zone.")