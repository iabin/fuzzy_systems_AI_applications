from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


MANEUVER_STOPPING = "Stopping"
MANEUVER_OVERTAKING = "Overtaking"
MANEUVER_U_TURNINGS = "U-Turnings"


def _mu_positive(value: float, threshold: float) -> float:
    """Membership for a positive condition, saturating at 1."""
    if threshold <= 0 or value <= 0:
        return 0.0
    return min(value / threshold, 1.0)


def _mu_negative(value: float, threshold_abs: float) -> float:
    """Membership for a negative condition, saturating at 1."""
    if threshold_abs <= 0 or value >= 0:
        return 0.0
    return min(abs(value) / threshold_abs, 1.0)


def _mu_linear_below(value: float, max_value: float) -> float:
    """Membership that is 1 at/below 0 and decreases to 0 at max_value."""
    if max_value <= 0:
        return 0.0
    return max(0.0, min((max_value - value) / max_value, 1.0))


def _mu_mid_range(value: float, low: float, high: float) -> float:
    """Triangular membership for intermediate values in [low, high]."""
    if value <= low or value >= high or high <= low:
        return 0.0
    center = (low + high) / 2.0
    half_width = (high - low) / 2.0
    return max(0.0, 1.0 - abs(value - center) / half_width)


def _mu_triangle_array(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership over an array domain."""
    if not (a < b < c):
        return np.zeros_like(x)
    y = np.zeros_like(x)
    left = (x >= a) & (x <= b)
    right = (x >= b) & (x <= c)
    y[left] = (x[left] - a) / (b - a)
    y[right] = (c - x[right]) / (c - b)
    return np.clip(y, 0.0, 1.0)


def _mu_trapezoid_array(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership over an array domain."""
    if not (a <= b <= c <= d):
        return np.zeros_like(x)
    y = np.zeros_like(x)

    if b > a:
        rise = (x >= a) & (x <= b)
        y[rise] = (x[rise] - a) / (b - a)
    else:
        y[x == a] = 1.0

    top = (x >= b) & (x <= c)
    y[top] = 1.0

    if d > c:
        fall = (x >= c) & (x <= d)
        y[fall] = (d - x[fall]) / (d - c)
    else:
        y[x == d] = 1.0

    return np.clip(y, 0.0, 1.0)


def _output_membership(label: str, x: np.ndarray, cfg: Any) -> np.ndarray:
    """Output fuzzy set memberships for maneuver confidence in [0, 1]."""
    if label == "low":
        return _mu_trapezoid_array(
            x,
            cfg.out_low_a,
            cfg.out_low_b,
            cfg.out_low_c,
            cfg.out_low_d,
        )
    if label == "medium":
        return _mu_triangle_array(x, cfg.out_mid_a, cfg.out_mid_b, cfg.out_mid_c)
    if label == "high":
        return _mu_trapezoid_array(
            x,
            cfg.out_high_a,
            cfg.out_high_b,
            cfg.out_high_c,
            cfg.out_high_d,
        )
    raise ValueError(f"Unsupported output fuzzy label: {label}")


def _trapz_area(y: np.ndarray, x: np.ndarray) -> float:
    """Numerical integration by trapezoidal rule."""
    if y.size < 2 or x.size < 2 or y.size != x.size:
        return 0.0
    dx = np.diff(x)
    avg_h = (y[:-1] + y[1:]) * 0.5
    return float(np.sum(avg_h * dx))


def _mamdani_centroid(rule_outputs: list[tuple[float, str]], cfg: Any) -> float:
    """Aggregate rule consequents with max-min Mamdani and centroid defuzzification."""
    n_points = int(getattr(cfg, "output_universe_points", 201))
    if n_points < 3:
        n_points = 201

    x = np.linspace(0.0, 1.0, n_points)
    agg = np.zeros_like(x)

    for strength, label in rule_outputs:
        alpha = max(0.0, min(float(strength), 1.0))
        if alpha <= 0.0:
            continue
        consequent = _output_membership(label, x, cfg)
        implied = np.minimum(alpha, consequent)
        agg = np.maximum(agg, implied)

    area = _trapz_area(agg, x)
    if area <= 1e-12:
        return 0.0
    centroid = float(_trapz_area(x * agg, x) / area)
    return max(0.0, min(centroid, 1.0))


def extract_window_measurements(window_df: pd.DataFrame, time_col: str = "Elapsed time") -> Dict[str, float]:
    """Extract base kinematic and control measurements from one window."""
    required = [time_col, "speed", "Brake pedal", "Gas pedal"]
    missing = [c for c in required if c not in window_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in window: {missing}")

    t = window_df[time_col]
    speed = window_df["speed"]
    brake = window_df["Brake pedal"]
    gas = window_df["Gas pedal"]
    steering = (
        window_df["Steering wheel angle"]
        if "Steering wheel angle" in window_df.columns
        else pd.Series(0.0, index=window_df.index)
    )

    dt = t.diff()
    dv = speed.diff()
    acc = (dv / dt).replace([float("inf"), float("-inf")], pd.NA).dropna()

    start_speed = float(speed.iloc[0])
    end_speed = float(speed.iloc[-1])

    return {
        "mean_acc": float(acc.mean()) if not acc.empty else 0.0,
        "start_speed": start_speed,
        "end_speed": end_speed,
        "speed_drop": start_speed - end_speed,
        "speed_rise": end_speed - start_speed,
        "mean_speed": float(speed.mean()),
        "mean_brake": float(brake.mean()),
        "brake_rise": float(brake.iloc[-1] - brake.iloc[0]),
        "mean_gas": float(gas.mean()),
        "mean_steering": float(steering.mean()) if not steering.empty else 0.0,
        "end_steering": float(steering.iloc[-1]) if not steering.empty else 0.0,
        "max_steering": float(steering.max()) if not steering.empty else 0.0,
        "mean_abs_steering": float(steering.abs().mean()),
        "steering_range": float(steering.max() - steering.min()) if not steering.empty else 0.0,
    }


def compute_fuzzy_memberships_stopping(meas: Dict[str, float], cfg: Any) -> Dict[str, float]:
    """Compute fuzzy antecedents for stopping detection."""
    mu_decelerating = _mu_negative(meas["mean_acc"], abs(cfg.strong_negative_accel))

    mu_start_not_zero = min(meas["start_speed"] / cfg.start_speed_not_near_zero, 1.0)
    mu_end_near_zero = _mu_linear_below(meas["end_speed"], cfg.end_speed_near_zero)
    mu_speed_drop = _mu_positive(meas["speed_drop"], cfg.min_speed_drop)
    mu_reaches_near_zero = min(mu_start_not_zero, mu_end_near_zero, mu_speed_drop)

    mu_high_brake = _mu_positive(meas["mean_brake"], cfg.high_brake_threshold)
    mu_brake_rise = _mu_positive(meas["brake_rise"], cfg.brake_rise_threshold)
    mu_low_gas = _mu_linear_below(meas["mean_gas"], cfg.low_gas_threshold)

    mu_mid_speed = _mu_mid_range(meas["mean_speed"], cfg.mid_speed_low, cfg.mid_speed_high)
    mu_intermediate_transition = min(mu_mid_speed, mu_decelerating, mu_brake_rise)

    mu_start_near_zero = _mu_linear_below(meas["start_speed"], cfg.start_speed_near_zero)

    return {
        "mu_decelerating": mu_decelerating,
        "mu_reaches_near_zero": mu_reaches_near_zero,
        "mu_high_brake": mu_high_brake,
        "mu_brake_rise": mu_brake_rise,
        "mu_intermediate_transition": mu_intermediate_transition,
        "mu_low_gas": mu_low_gas,
        "mu_start_near_zero": mu_start_near_zero,
    }


def compute_stopping_score(mu: Dict[str, float], cfg: Any) -> float:
    """Compute stopping score using Mamdani aggregation + centroid defuzzification."""
    rule_outputs = [
        (cfg.w_rule_decelerating * mu["mu_decelerating"], "high"),
        (cfg.w_rule_reaches_near_zero_bonus * mu["mu_reaches_near_zero"], "high"),
        (cfg.w_rule_high_brake * mu["mu_high_brake"], "high"),
        (cfg.w_rule_brake_rise * mu["mu_brake_rise"], "medium"),
        (cfg.w_rule_intermediate_transition * mu["mu_intermediate_transition"], "medium"),
        (cfg.w_rule_low_gas * mu["mu_low_gas"], "medium"),
        (cfg.start_near_zero_penalty_weight * mu["mu_start_near_zero"], "low"),
    ]
    return _mamdani_centroid(rule_outputs, cfg)


def compute_fuzzy_memberships_overtaking(meas: Dict[str, float], cfg: Any) -> Dict[str, float]:
    """Compute fuzzy antecedents for overtaking detection."""
    mu_accelerating = _mu_positive(meas["mean_acc"], cfg.strong_positive_accel)
    mu_speed_rise = _mu_positive(meas["speed_rise"], cfg.min_speed_rise)
    mu_mean_speed_high = _mu_positive(meas["mean_speed"], cfg.min_mean_speed_for_overtaking)
    mu_high_gas = _mu_positive(meas["mean_gas"], cfg.high_gas_threshold)
    mu_low_brake = _mu_linear_below(meas["mean_brake"], cfg.low_brake_for_overtaking)
    mu_high_brake = _mu_positive(meas["mean_brake"], cfg.high_brake_for_overtaking_penalty)
    steering_activity = 0.6 * meas["mean_abs_steering"] + 0.4 * meas["steering_range"]
    mu_steering_activity = _mu_mid_range(steering_activity, cfg.steering_activity_min, cfg.steering_activity_max)

    return {
        "mu_accelerating": mu_accelerating,
        "mu_speed_rise": mu_speed_rise,
        "mu_mean_speed_high": mu_mean_speed_high,
        "mu_high_gas": mu_high_gas,
        "mu_low_brake": mu_low_brake,
        "mu_high_brake": mu_high_brake,
        "mu_steering_activity": mu_steering_activity,
    }


def compute_overtaking_score(mu: Dict[str, float], cfg: Any) -> float:
    """Compute overtaking score using Mamdani aggregation + centroid defuzzification."""
    rule_outputs = [
        (cfg.w_rule_accelerating * mu["mu_accelerating"], "high"),
        (cfg.w_rule_speed_rise * mu["mu_speed_rise"], "high"),
        (cfg.w_rule_mean_speed_high * mu["mu_mean_speed_high"], "medium"),
        (cfg.w_rule_high_gas * mu["mu_high_gas"], "high"),
        (cfg.w_rule_low_brake * mu["mu_low_brake"], "medium"),
        (cfg.w_rule_high_brake_penalty * mu["mu_high_brake"], "low"),
        (cfg.w_rule_steering_activity * mu["mu_steering_activity"], "medium"),
    ]
    return _mamdani_centroid(rule_outputs, cfg)


def compute_fuzzy_memberships_u_turnings(meas: Dict[str, float], cfg: Any) -> Dict[str, float]:
    """Compute fuzzy antecedents for U-turnings detection."""
    mu_mean_abs_steering_high = _mu_positive(meas["mean_abs_steering"], cfg.u_turn_mean_abs_steering_threshold)
    mu_steering_range_high = _mu_positive(meas["steering_range"], cfg.u_turn_steering_range_threshold)

    return {
        "mu_mean_abs_steering_high": mu_mean_abs_steering_high,
        "mu_steering_range_high": mu_steering_range_high,
    }


def compute_u_turnings_score(mu: Dict[str, float], cfg: Any) -> float:
    """Compute U-turnings score using Mamdani aggregation + centroid defuzzification."""
    rule_outputs = [
        (cfg.w_rule_u_turn_mean_abs_steering * mu["mu_mean_abs_steering_high"], "high"),
        (cfg.w_rule_u_turn_steering_range * mu["mu_steering_range_high"], "high"),
    ]
    return _mamdani_centroid(rule_outputs, cfg)


def is_maneuver_window(
    window_df: pd.DataFrame,
    maneuver_name: str,
    cfg: Any,
    time_col: str = "Elapsed time",
    prev_true: bool = False,
) -> bool:
    """Return True when a window satisfies the maneuver-specific logic."""
    meas = extract_window_measurements(window_df, time_col=time_col)

    if maneuver_name == MANEUVER_STOPPING:
        mu = compute_fuzzy_memberships_stopping(meas, cfg)
        score = compute_stopping_score(mu, cfg)
        return score >= cfg.stopping_score_threshold

    if maneuver_name == MANEUVER_OVERTAKING:
        mu = compute_fuzzy_memberships_overtaking(meas, cfg)
        score = compute_overtaking_score(mu, cfg)
        return score >= cfg.overtaking_score_threshold

    if maneuver_name == MANEUVER_U_TURNINGS:
        mu = compute_fuzzy_memberships_u_turnings(meas, cfg)
        score = compute_u_turnings_score(mu, cfg)
        return score >= cfg.u_turn_score_threshold

    raise ValueError(f"Unsupported maneuver: {maneuver_name}")
