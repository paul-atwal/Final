import pandas as pd

# Cache variable
_df = None

def load_nflfastR_data(path="data/2007_to_2024_nflfastR.csv.gz"):
    global _df
    if _df is None:
        print("Loading nflfastR data...")
        _df = pd.read_csv(path, compression="gzip", low_memory=False)
    return _df

# Additional code for preprocessing data
SOURCE_PATH = "data/2007_to_2024_nflfastR.csv.gz"

def main():
    print("Loading full dataset...")
    df = pd.read_csv(SOURCE_PATH, compression="gzip", low_memory=False)

    # --- WP model relevant data ---
    print("Saving wp_model data...")
    wp_df = df[
        df["wp"].notna() &
        df["posteam"].notna() &
        df["defteam"].notna() &
        df["yardline_100"].notna() & 
        df["down"].notna()
    ].copy()
    wp_df = wp_df[
        ["down", "yardline_100", "ydstogo", "qtr", "quarter_seconds_remaining",
         "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "wp"]
    ]
    wp_df.to_pickle("data/wp_model_data.pkl")

    # --- FG model relevant data ---
    print("Saving fg_model data...")
    fg_df = df[df["play_type"] == "field_goal"].copy()
    fg_df = fg_df.dropna(subset=["field_goal_result", "yardline_100"])
    fg_df = fg_df[["yardline_100", "field_goal_result"]]
    fg_df.to_pickle("data/fg_model_data.pkl")

    # --- Punt model relevant data ---
    print("Saving punt_model data...")
    punt_df = df[df["play_type"] == "punt"].copy()
    punt_df = punt_df[["yardline_100", "kick_distance", "return_yards"]]
    punt_df.to_pickle("data/punt_model_data.pkl")

    # --- Go model relevant data ---
    print("Saving go_model data...")
    go_df = df[
        (df["down"] == 4) &
        (df["play_type"].isin(["run", "pass"]))
    ].copy()
    go_df = go_df.dropna(subset=["ydstogo", "yardline_100", "yards_gained"])
    go_df = go_df[["yardline_100", "ydstogo", "yards_gained"]]
    go_df.to_pickle("data/go_model_data.pkl")

    # --- Full dataset pickle ---
    print("Saving full dataset...")
    df.to_pickle("data/full_data.pkl")

    print("Done preprocessing.")

if __name__ == "__main__":
    main()