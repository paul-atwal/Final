from wp_model import predict_win_probability
from fg_model import predict_fg_success_probability
from punt_model import predict_opponent_yardline
from go_model import predict_conversion_probability

def flip_possession(state):
    state["score_differential"] = -state["score_differential"]
    state["posteam_timeouts_remaining"], state["defteam_timeouts_remaining"] = (
        state["defteam_timeouts_remaining"],
        state["posteam_timeouts_remaining"]
    )
    return state

def simulate_post_score_state(state, added_points):
    state = state.copy()
    state["score_differential"] += added_points
    state["yardline_100"] = 75  # Opponent starts at their 25
    state["ydstogo"] = 10
    state["down"] = 1
    return flip_possession(state)

def simulate_decision(game_state):
    """
    Given game state, simulate expected WP for go for it, field goal, and punt.
    game_state should be a dictionary with:
    - yardline_100
    - ydstogo
    - score_differential
    - posteam_timeouts_remaining
    - defteam_timeouts_remaining
    - down
    """
    y = game_state["yardline_100"]
    t = game_state["ydstogo"]
    game_state = game_state.copy()
    game_state["down"] = 4

    is_goal_to_go = y <= t

    # Go for it 
    p_go_success = predict_conversion_probability(y, t)

    if is_goal_to_go:
        go_success_state = simulate_post_score_state(game_state, 7)
        wp_go_success = 1 - predict_win_probability(go_success_state)
    else:
        go_success_state = game_state.copy()
        go_success_state["yardline_100"] = y - t
        go_success_state["ydstogo"] = 10
        go_success_state["down"] = 1
        wp_go_success = predict_win_probability(go_success_state)

    go_fail_state = game_state.copy()
    go_fail_state["yardline_100"] = 100 - y
    go_fail_state["ydstogo"] = (10 if go_fail_state["yardline_100"] > 10 else go_fail_state["yardline_100"])
    go_fail_state["down"] = 1
    go_fail_state = flip_possession(go_fail_state)
    wp_go_fail = 1 - predict_win_probability(go_fail_state)


    expected_wp_go = p_go_success * wp_go_success + (1 - p_go_success) * wp_go_fail

    # Field Goal
    p_fg_success = predict_fg_success_probability(y)
    fg_success_state = simulate_post_score_state(game_state, 3)
    wp_fg_success = 1 - predict_win_probability(fg_success_state)

    fg_fail_state = game_state.copy()
    if y <= 20:
        fg_fail_state["yardline_100"] = 100 - y
    else:
        fg_fail_state["yardline_100"] = 100 - (y + 8)
    fg_fail_state["ydstogo"] = 10
    fg_fail_state["down"] = 1
    fg_fail_state = flip_possession(fg_fail_state)
    wp_fg_fail = 1 - predict_win_probability(fg_fail_state)

    expected_wp_fg = p_fg_success * wp_fg_success + (1 - p_fg_success) * wp_fg_fail

    # Punt
    if y >= 35:
        opp_yardline = predict_opponent_yardline(y)
        punt_state = game_state.copy()
        punt_state["yardline_100"] = opp_yardline
        punt_state["ydstogo"] = 10
        punt_state["down"] = 1
        punt_state = flip_possession(punt_state)
        wp_punt = 1 - predict_win_probability(punt_state)
    else:
        wp_punt = 0 

    return {
        "go": expected_wp_go,
        "field_goal": expected_wp_fg,
        "punt": wp_punt
    }
