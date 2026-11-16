"""
CS238: Final Project
This file generates an initial-state synthetic dataset of post-flood disaster scenarios.

Each row represents one scenario --> the true day-0 state (community needs, road
conditions, resources) plus the agency's nosiy day-0 reports, which are
accurate with probability OBS_ACCURACY.

The severe level parameters are drawn from the September 2024 Nepal floods
(by 1 Oct 2024 only 9 of 37 national highways were open, ~75% closed;
our "severe" road distribution has 80% of roads damaged or impassable
as a result).

Output: synthetic_disasters.csv
"""
import numpy as np
import pandas as pd

SEED = 0  # to reproduce dataset
EPISODES_PER_SEVERITY_LEVEL = 150
OBS_ACCURACY = 0.8
COMMUNITIES = [f"C{i}" for i in range(1, 6)]
NEED_LEVELS = [0, 1, 2, 3]
ROAD_STATES = ["intact", "damaged", "impassable"]
ROADS = [
    ("Depot", "C1"),
    ("Depot", "C2"),
    ("C1", "C3"),
    ("C1", "C4"),
    ("C1", "C5"),
    ("C2", "C4"),
    ("C2", "C5"),
]
ROAD_NAMES = [f"{a}_{b}" for a, b in ROADS]
POPULATION_WEIGHTS = [1, 2, 1, 3, 2]
INITIAL_RESOURCES = {"supplies": 15, "materials": 10, "budget": 25}
SEVERITY_PARAMS = {
    "mild": {
        "needs": [0.40, 0.40, 0.15, 0.05],
        "roads": [0.80, 0.15, 0.05],
        "need_growth_prob": 0.2,
    },
    "moderate": {
        "needs": [0.10, 0.30, 0.40, 0.20],
        "roads": [0.50, 0.35, 0.15],
        "need_growth_prob": 0.2,
    },
    "severe": {
        "needs": [0.05, 0.10, 0.35, 0.50],
        "roads": [0.20, 0.40, 0.40],
        "need_growth_prob": 0.2,
    },
}


def noisy(rng, true_value, values, accuracy=OBS_ACCURACY):
    """Report the true value w.p. `accuracy`, else a uniformly random wrong value."""
    if rng.random() < accuracy:
        return true_value
    return rng.choice([v for v in values if v != true_value])


def generate(seed=SEED, episodes_per_severity=EPISODES_PER_SEVERITY_LEVEL):
    ran_num_gen = np.random.default_rng(seed)
    rows = []

    for severity, params in SEVERITY_PARAMS.items():
        for i in range(episodes_per_severity):
            needs = ran_num_gen.choice(NEED_LEVELS, size=len(COMMUNITIES), p=params["needs"])
            conditions = ran_num_gen.choice(ROAD_STATES, size=len(ROADS), p=params["roads"])

            row = {
                "episode": len(rows),
                "severity": severity,
                "need_growth_prob": params["need_growth_prob"],
                "obs_accuracy": OBS_ACCURACY,
                "initial_supplies": INITIAL_RESOURCES["supplies"],
                "initial_materials": INITIAL_RESOURCES["materials"],
                "initial_budget": INITIAL_RESOURCES["budget"],
            }

            for j, c in enumerate(COMMUNITIES):
                row[f"{c}_weight"] = POPULATION_WEIGHTS[j]
                row[f"{c}_needs"] = int(needs[j])
                row[f"{c}_needs_obs"] = int(noisy(ran_num_gen, int(needs[j]), NEED_LEVELS))

            for j, name in enumerate(ROAD_NAMES):
                row[f"{name}_condition"] = str(conditions[j])
                row[f"{name}_condition_obs"] = str(noisy(ran_num_gen, str(conditions[j]), ROAD_STATES))

            rows.append(row)

    return pd.DataFrame(rows)


def summarize(df):
    """Per-severity summary: mean true need, observation accuracy, road damage."""
    need_cols = [f"{c}_needs" for c in COMMUNITIES]
    obs_cols = [f"{c}_needs_obs" for c in COMMUNITIES]
    road_cols = [f"{r}_condition" for r in ROAD_NAMES]
    road_obs_cols = [f"{r}_condition_obs" for r in ROAD_NAMES]

    out = pd.DataFrame({
        "severity": df["severity"],
        "mean_true_need": df[need_cols].mean(axis=1),
        "weighted_true_need": sum(df[f"{c}_needs"] * df[f"{c}_weight"] for c in COMMUNITIES),
        "pct_roads_damaged": (df[road_cols] == "damaged").mean(axis=1) * 100,
        "pct_roads_impassable": (df[road_cols] == "impassable").mean(axis=1) * 100,
        "need_report_accuracy": (df[need_cols].values == df[obs_cols].values).mean(axis=1) * 100,
        "road_report_accuracy": (df[road_cols].values == df[road_obs_cols].values).mean(axis=1) * 100,
    })
    order = list(SEVERITY_PARAMS)
    return (out.groupby("severity").mean().reindex(order).round(2))


if __name__ == "__main__":
    df = generate()
    df.to_csv("synthetic_disasters.csv", index=False)
    print(f"Just successfully saved {len(df)} scenarios to synthetic_disasters.csv\n")
    print(summarize(df).to_string())