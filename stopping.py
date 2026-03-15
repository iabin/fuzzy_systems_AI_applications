from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

import pandas as pd

from fuzzy_rules import is_maneuver_window


DEFAULT_FILE = Path("data/Driver1/STISIMData_Stopping.xlsx")
TIME_COL = "Elapsed time"
LABEL_COL = "Maneuver marker flag"
MANEUVER_STOPPING = "Stopping"
MANEUVER_OVERTAKING = "Overtaking"
MANEUVER_U_TURNINGS = "U-Turnings"


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for temporal window generation."""

    window_seconds: float = 1.0
    step_seconds: float = 0.25
    min_samples: int = 2


@dataclass(frozen=True)
class FuzzyRuleConfig:
    """Configuration for fuzzy rule thresholds and weights."""

    strong_negative_accel: float = -0.5
    start_speed_not_near_zero: float = 10.0
    start_speed_near_zero: float = 2.0
    mid_speed_low: float = 8.0
    mid_speed_high: float = 35.0
    end_speed_near_zero: float = 3.0
    min_speed_drop: float = 5.0
    high_brake_threshold: float = 3000.0
    brake_rise_threshold: float = 2500.0
    low_gas_threshold: float = 5000.0

    w_rule_decelerating: float = 0.30
    w_rule_reaches_near_zero_bonus: float = 0.08
    w_rule_high_brake: float = 0.55
    w_rule_brake_rise: float = 0.35
    w_rule_intermediate_transition: float = 0.25
    w_rule_low_gas: float = 0.28

    start_near_zero_penalty_weight: float = 0.35
    stopping_score_threshold: float = 0.70

    strong_positive_accel: float = 0.50
    min_speed_rise: float = 4.8
    min_mean_speed_for_overtaking: float = 21.5
    high_gas_threshold: float = 12500.0
    low_brake_for_overtaking: float = 1100.0
    high_brake_for_overtaking_penalty: float = 2500.0
    steering_activity_min: float = 5.5
    steering_activity_max: float = 55.0

    w_rule_accelerating: float = 0.34
    w_rule_speed_rise: float = 0.33
    w_rule_mean_speed_high: float = 0.24
    w_rule_high_gas: float = 0.28
    w_rule_low_brake: float = 0.24
    w_rule_high_brake_penalty: float = 0.30
    w_rule_steering_activity: float = 0.22
    overtaking_score_threshold: float = 0.67

    steering_left_threshold: float = -2.0
    steering_right_threshold: float = 2.0
    steering_forward_abs_threshold: float = 1.5
    overtaking_min_mean_accel: float = -0.05
    overtaking_min_speed_rise: float = -0.50
    overtaking_medium_heavy_accel_threshold: float = 0.30
    overtaking_exit_forward_windows: int = 2
    overtaking_exit_accel_threshold: float = 0.1

    u_turn_mean_abs_steering_threshold: float = 33.0
    u_turn_steering_range_threshold: float = 75.0
    w_rule_u_turn_mean_abs_steering: float = 0.62
    w_rule_u_turn_steering_range: float = 0.58
    u_turn_score_threshold: float = 0.79

    # Mamdani output fuzzy sets over maneuver confidence in [0, 1]
    output_universe_points: int = 201
    out_low_a: float = 0.0
    out_low_b: float = 0.0
    out_low_c: float = 0.35
    out_low_d: float = 0.55
    out_mid_a: float = 0.30
    out_mid_b: float = 0.50
    out_mid_c: float = 0.70
    out_high_a: float = 0.55
    out_high_b: float = 0.75
    out_high_c: float = 1.0
    out_high_d: float = 1.0


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level configuration for timeline inference and voting."""

    window: WindowConfig = WindowConfig()
    fuzzy: FuzzyRuleConfig = FuzzyRuleConfig()
    vote_threshold: float = 0.25


def load_excel(file_path: str | Path) -> pd.DataFrame:
    """Load one simulator file into a DataFrame."""
    return pd.read_excel(file_path)


def sort_by_time(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """Return a copy sorted by time column."""
    if time_col not in df.columns:
        raise ValueError(f"Missing required column: {time_col}")
    return df.sort_values(time_col).reset_index(drop=True)


def split_into_windows(df: pd.DataFrame, config: WindowConfig, time_col: str = TIME_COL) -> List[pd.DataFrame]:
    """Split a sorted DataFrame into fixed-length sliding windows."""
    if config.window_seconds <= 0 or config.step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be > 0")
    if time_col not in df.columns:
        raise ValueError(f"Missing required column: {time_col}")

    t0 = float(df[time_col].min())
    t1 = float(df[time_col].max())
    windows: List[pd.DataFrame] = []

    start = t0
    while start + config.window_seconds <= t1 + 1e-9:
        end = start + config.window_seconds
        mask = (df[time_col] >= start) & (df[time_col] < end)
        win = df.loc[mask, :].copy()
        if len(win) >= config.min_samples:
            windows.append(win)
        start += config.step_seconds

    return windows


def split_into_non_overlapping_windows(
    df: pd.DataFrame,
    config: WindowConfig,
    time_col: str = TIME_COL,
) -> List[pd.DataFrame]:
    """Split a sorted DataFrame into fixed-length non-overlapping windows."""
    non_overlap_cfg = WindowConfig(
        window_seconds=config.window_seconds,
        step_seconds=config.window_seconds,
        min_samples=config.min_samples,
    )
    return split_into_windows(df, non_overlap_cfg, time_col=time_col)


def classify_windows_maneuver(
    windows: List[pd.DataFrame],
    maneuver_name: str,
    cfg: FuzzyRuleConfig = FuzzyRuleConfig(),
) -> List[bool]:
    """Classify each window as maneuver/non-maneuver."""
    if maneuver_name == MANEUVER_OVERTAKING:
        preds: List[bool] = []
        prev_true = False
        for window_df in windows:
            current = is_maneuver_window(
                window_df,
                maneuver_name,
                cfg,
                time_col=TIME_COL,
                prev_true=prev_true,
            )
            preds.append(current)
            prev_true = current
        return preds

    return [is_maneuver_window(w, maneuver_name, cfg, time_col=TIME_COL) for w in windows]


def aggregate_overlapping_votes(
    sorted_df: pd.DataFrame,
    windows: List[pd.DataFrame],
    predictions: List[bool],
    vote_threshold: float,
    time_col: str = TIME_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """
    Resolve overlap conflicts by sample-wise majority voting.

    A sample is predicted as maneuver if:
    positive_votes / total_votes >= vote_threshold.
    """
    votes = pd.Series(0, index=sorted_df.index, dtype="int64")
    totals = pd.Series(0, index=sorted_df.index, dtype="int64")

    for win, pred in zip(windows, predictions):
        idx = win.index
        totals.loc[idx] += 1
        if pred:
            votes.loc[idx] += 1

    has_votes = totals > 0
    ratio = votes / totals.where(has_votes, 1)
    pred = (has_votes & (ratio >= vote_threshold)).astype(int)

    out = pd.DataFrame(
        {
            "time": sorted_df[time_col].astype(float),
            "pred_vote_ratio": ratio.astype(float),
            "pred_maneuver": pred.astype(int),
        }
    )
    if label_col in sorted_df.columns:
        out["gt_maneuver"] = sorted_df[label_col].astype(int)
    return out


def plot_maneuver_timeline(timeline_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot predicted vs marker maneuver for one timeline."""
    import matplotlib.pyplot as plt

    out = Path(output_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timeline_df["time"], timeline_df["pred_maneuver"], label="Predicted maneuver", linewidth=1.5)
    if "gt_maneuver" in timeline_df.columns:
        ax.plot(timeline_df["time"], timeline_df["gt_maneuver"], label="Marker flag maneuver", linewidth=1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Maneuver (0/1)")
    ax.set_title("Window Maneuver Prediction vs Ground Truth")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def list_maneuver_files(maneuver_name: str, data_root: str | Path = "data") -> List[Path]:
    """Return all maneuver files under Driver folders."""
    return sorted(Path(data_root).glob(f"Driver*/STISIMData_{maneuver_name}.xlsx"))


def list_stopping_files(data_root: str | Path = "data") -> List[Path]:
    """Backward-compatible wrapper for stopping file discovery."""
    return list_maneuver_files(MANEUVER_STOPPING, data_root)


def _driver_label_from_path(path: Path) -> str:
    """Extract user-friendly driver label from path."""
    match = re.search(r"Driver(\d+)", str(path))
    return f"Driver {match.group(1)}" if match else path.parent.name


def infer_timeline_for_file(file_path: Path, maneuver_name: str, cfg: InferenceConfig) -> pd.DataFrame:
    """Full inference pipeline for one file: load -> sort -> windows -> predict -> vote timeline."""
    raw_df = load_excel(file_path)
    sorted_df = sort_by_time(raw_df, TIME_COL)

    windows = split_into_windows(sorted_df, cfg.window, TIME_COL)
    compare_windows = windows
    if maneuver_name == MANEUVER_OVERTAKING:
        # Overtaking previous-window logic uses non-overlapping windows by request.
        compare_windows = split_into_non_overlapping_windows(sorted_df, cfg.window, TIME_COL)

    preds = classify_windows_maneuver(compare_windows, maneuver_name, cfg.fuzzy)
    # For overtaking, any positive window vote is enough to mark maneuver.
    effective_vote_threshold = 1e-9 if maneuver_name == MANEUVER_OVERTAKING else cfg.vote_threshold
    timeline = aggregate_overlapping_votes(sorted_df, compare_windows, preds, vote_threshold=effective_vote_threshold)
    timeline["driver"] = _driver_label_from_path(file_path)
    return timeline


def plot_mega_maneuver_graph(
    timelines: List[pd.DataFrame],
    maneuver_name: str,
    output_path: str | Path,
) -> Path:
    """Plot one stacked subplot per driver timeline in a single mega-graph."""
    import matplotlib.pyplot as plt

    if not timelines:
        raise ValueError("No timelines provided to plot")

    out = Path(output_path)
    n = len(timelines)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, df in zip(axes, timelines):
        driver = str(df["driver"].iloc[0]) if "driver" in df.columns and not df.empty else "Driver"
        ax.plot(df["time"], df["pred_maneuver"], label="Predicted maneuver", linewidth=1.4)
        if "gt_maneuver" in df.columns:
            ax.plot(df["time"], df["gt_maneuver"], label="Marker flag maneuver", linewidth=1.0)
        ax.set_ylabel("Maneuver")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"{driver} - {maneuver_name} Identified")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{maneuver_name} Maneuver Identification Across All Drivers", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_mega_stopping_graph(timelines: List[pd.DataFrame], output_path: str | Path = "stopping_megagraph.png") -> Path:
    """Backward-compatible stopping mega-graph wrapper."""
    return plot_mega_maneuver_graph(timelines, MANEUVER_STOPPING, output_path)


def build_default_inference_config(window_seconds: float = 1.0, overlap_ratio: float = 0.75) -> InferenceConfig:
    """Create a default inference config from intuitive overlap settings."""
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")
    step_seconds = window_seconds * (1.0 - overlap_ratio)
    return InferenceConfig(window=WindowConfig(window_seconds=window_seconds, step_seconds=step_seconds, min_samples=2))


def run_all_drivers_maneuver(
    maneuver_name: str,
    data_root: str | Path = "data",
    cfg: InferenceConfig | None = None,
    mega_plot_path: str | Path | None = None,
) -> Path:
    """Run maneuver identification for all drivers and save one mega-graph."""
    effective_cfg = cfg or build_default_inference_config(window_seconds=1.0, overlap_ratio=0.75)
    files = list_maneuver_files(maneuver_name, data_root)
    if not files:
        raise FileNotFoundError(f"No {maneuver_name} files found under data/Driver*/STISIMData_{maneuver_name}.xlsx")

    if mega_plot_path is None:
        mega_plot_path = f"{maneuver_name.lower()}_megagraph.png"

    timelines: List[pd.DataFrame] = []
    for file_path in files:
        timeline = infer_timeline_for_file(file_path, maneuver_name, effective_cfg)
        timelines.append(timeline)
        print(f"processed: {file_path} rows={len(timeline)} pred_ones={int(timeline['pred_maneuver'].sum())}")

    out = plot_mega_maneuver_graph(timelines, maneuver_name, output_path=mega_plot_path)
    return out


def run_all_drivers_stopping(
    data_root: str | Path = "data",
    cfg: InferenceConfig | None = None,
    mega_plot_path: str | Path = "stopping_megagraph.png",
) -> Path:
    """Backward-compatible wrapper for stopping identification."""
    return run_all_drivers_maneuver(
        MANEUVER_STOPPING,
        data_root=data_root,
        cfg=cfg,
        mega_plot_path=mega_plot_path,
    )


def run_all_drivers_overtaking(
    data_root: str | Path = "data",
    cfg: InferenceConfig | None = None,
    mega_plot_path: str | Path = "overtaking_megagraph.png",
) -> Path:
    """Run overtaking identification for all drivers and save one mega-graph."""
    effective_cfg = cfg or build_default_inference_config(window_seconds=1.0, overlap_ratio=0.0)
    return run_all_drivers_maneuver(
        MANEUVER_OVERTAKING,
        data_root=data_root,
        cfg=effective_cfg,
        mega_plot_path=mega_plot_path,
    )


def run_all_drivers_u_turnings(
    data_root: str | Path = "data",
    cfg: InferenceConfig | None = None,
    mega_plot_path: str | Path = "u_turnings_megagraph.png",
) -> Path:
    """Run U-turnings identification for all drivers and save one mega-graph."""
    return run_all_drivers_maneuver(
        MANEUVER_U_TURNINGS,
        data_root=data_root,
        cfg=cfg,
        mega_plot_path=mega_plot_path,
    )


if __name__ == "__main__":
    stopping_plot = run_all_drivers_stopping(data_root="data", mega_plot_path="stopping_megagraph.png")
    print("stopping_mega_plot_saved:", str(stopping_plot))

    overtaking_plot = run_all_drivers_overtaking(data_root="data", mega_plot_path="overtaking_megagraph.png")
    print("overtaking_mega_plot_saved:", str(overtaking_plot))

    u_turnings_plot = run_all_drivers_u_turnings(data_root="data", mega_plot_path="u_turnings_megagraph.png")
    print("u_turnings_mega_plot_saved:", str(u_turnings_plot))
