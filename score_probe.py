import pandas as pd

import fuzzy_rules as fr
import stopping

cfg = stopping.FuzzyRuleConfig()
files = {
    "Stopping": "data/Driver1/STISIMData_Stopping.xlsx",
    "Overtaking": "data/Driver1/STISIMData_Overtaking.xlsx",
    "U-Turnings": "data/Driver1/STISIMData_U-Turnings.xlsx",
}

for maneuver, path in files.items():
    df = stopping.sort_by_time(stopping.load_excel(path))
    windows = stopping.split_into_windows(
        df,
        stopping.WindowConfig(window_seconds=1.0, step_seconds=0.25, min_samples=2),
    )

    scores = []
    for w in windows:
        meas = fr.extract_window_measurements(w)
        if maneuver == "Stopping":
            mu = fr.compute_fuzzy_memberships_stopping(meas, cfg)
            score = fr.compute_stopping_score(mu, cfg)
        elif maneuver == "Overtaking":
            mu = fr.compute_fuzzy_memberships_overtaking(meas, cfg)
            score = fr.compute_overtaking_score(mu, cfg)
        else:
            mu = fr.compute_fuzzy_memberships_u_turnings(meas, cfg)
            score = fr.compute_u_turnings_score(mu, cfg)
        scores.append(score)

    s = pd.Series(scores)
    print(
        maneuver,
        "count",
        len(s),
        "min",
        round(float(s.min()), 4),
        "p50",
        round(float(s.quantile(0.5)), 4),
        "p75",
        round(float(s.quantile(0.75)), 4),
        "p90",
        round(float(s.quantile(0.9)), 4),
        "p95",
        round(float(s.quantile(0.95)), 4),
        "max",
        round(float(s.max()), 4),
    )
