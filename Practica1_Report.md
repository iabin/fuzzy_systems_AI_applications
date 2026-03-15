# Advanced IA Practice 1 Report

## 1. Context and Goal

This project implements maneuver identification from simulator telemetry using an interpretable fuzzy and rule-based approach, aligned with the practice intention of explainability.

The practical objective has been to detect driving maneuvers from windowed time series data and compare predictions against the provided marker labels.

Implemented maneuvers:

- Stopping
- Overtaking
- U-Turnings

Generated outputs:

- stopping_megagraph.png
- overtaking_megagraph.png
- u_turnings_megagraph.png

Main implementation file:

- stopping.py

## 2. Dataset and Structure

Workspace structure used:

- data/Driver1 ... data/Driver5
- Each driver folder contains maneuver-specific Excel files

Relevant files processed:

- STISIMData_Stopping.xlsx
- STISIMData_Overtaking.xlsx
- STISIMData_U-Turnings.xlsx

Core columns used in the final pipeline:

- Elapsed time
- speed
- Brake pedal
- Gas pedal
- Steering wheel angle
- Maneuver marker flag

Important practical observation:

- Labels are useful but not perfectly clean in temporal boundaries.
- Because of this, evaluation has focused on whether predictions fall inside maneuver regions rather than exact boundary matching.

## 3. Methodology Overview

The implementation follows a common sequence for all maneuvers:

1. Load Excel telemetry.
2. Sort by time.
3. Build temporal windows.
4. Extract window-level features.
5. Apply maneuver logic (fuzzy or rule-based).
6. Aggregate predictions to timeline.
7. Plot predicted vs marker curves.
8. Build all-driver mega-graph.

Design principles applied:

- Interpretability first.
- Explicit thresholds and weights.
- Maneuver-specific logic with shared pipeline.
- Easy parameter tuning.

## 4. Windowing and Aggregation

### 4.1 Windowing

Base default window:

- Window length: 1.0 s
- Configurable overlap via step size

Special choice for overtaking:

- Non-overlapping windows for previous-window rule logic
- This was explicitly requested to preserve temporal causality in sequence rules

### 4.2 Aggregation

Aggregation is done sample-wise via vote ratio from windows that cover each sample.

- pred_vote_ratio = positive_votes / total_votes
- pred_maneuver = 1 if vote ratio passes threshold

Special threshold behavior:

- Overtaking uses a near-zero vote threshold to enforce: if any rule window is true, prediction is treated as true at covered samples.
- Stopping and U-Turnings use the standard vote threshold from config.

## 5. Feature Extraction per Window

For each window, the following measurements are computed:

- mean_acc
- start_speed
- end_speed
- speed_drop
- speed_rise
- mean_speed
- mean_brake
- brake_rise
- mean_gas
- mean_steering
- end_steering
- max_steering
- mean_abs_steering
- steering_range

These are the interpretability bridge between raw telemetry and maneuver decisions.

## 6. Fuzzy Logic Foundations

The code uses explicit membership-like functions:

- Positive membership with saturation
- Negative membership with saturation
- Linear-below membership
- Mid-range triangular membership

This keeps decisions explainable while preserving fuzzy reasoning behavior.

General form:

- Membership values are in [0, 1]
- Rule score is a weighted combination
- Final decision by threshold comparison

## 7. Stopping: Final Fuzzy Rule System

Stopping remains the most complete fuzzy subsystem and best reflects the practice intention.

### 7.1 Fuzzy Antecedents

Computed memberships:

- mu_decelerating
- mu_reaches_near_zero
- mu_high_brake
- mu_brake_rise
- mu_intermediate_transition
- mu_low_gas
- mu_start_near_zero (penalty)

### 7.2 Rule Weights

Positive evidence weights:

- decelerating: 0.30
- reaches near zero bonus: 0.08
- high brake: 0.55
- brake rise: 0.35
- intermediate transition: 0.25
- low gas: 0.28

Penalty:

- start near zero penalty: 0.35

### 7.3 Stopping Score

Score structure:

- stopping_score = clamp(positive_sum - penalty, 0, 1)

Decision threshold:

- stopping_score >= 0.85

Interpretation:

- A window is stopping when braking and deceleration evidence are strong, low-gas behavior is consistent, and the segment does not start already near full stop.

## 8. Overtaking: Rule-Guided Hybrid (Final State)

Overtaking logic evolved through multiple iterations to satisfy explicit constraints requested during development.

### 8.1 Why Hybrid Instead of Pure Weighted Fuzzy

A pure weighted fuzzy overtaking score existed in code, but the final active behavior was adapted to user-required temporal start logic.

The final active overtaking rule is sequence-aware and intentionally strict about maneuver starts.

### 8.2 Active Overtaking Rule

A window starts overtaking when all are true:

- Steering state is left (using corrected simulator sign convention)
- Previous window state is false
- Mild motion consistency condition:
  - speed_rise >= overtaking_min_speed_rise
  OR
  - mean_acc >= overtaking_min_mean_accel

Additional enabling rule:

- Left turn with medium-heavy acceleration:
  - mean_acc >= overtaking_medium_heavy_accel_threshold

Current key thresholds:

- overtaking_min_mean_accel = -0.05
- overtaking_min_speed_rise = -0.50
- overtaking_medium_heavy_accel_threshold = 0.30

### 8.3 Overtaking Temporal Policy

- Non-overlapping windows for rule sequencing
- Near-any-vote aggregation behavior for overtaking timeline coverage

Interpretation:

- Overtaking detection is intentionally start-event sensitive, then projected to timeline through the chosen voting policy.

## 9. U-Turnings: Heavy Steering Rule Set

U-Turnings was added as a dedicated maneuver with a direct and interpretable rule system.

### 9.1 Active U-Turning Rule

A window is U-turning if either is true:

- mean_abs_steering >= u_turn_mean_abs_steering_threshold
- steering_range >= u_turn_steering_range_threshold

Final strict thresholds after tuning:

- u_turn_mean_abs_steering_threshold = 20.0
- u_turn_steering_range_threshold = 45.0

### 9.2 Rationale

U-turns are dominated by strong steering behavior and high angular travel in short intervals.

Using both average absolute angle and range improves robustness:

- mean_abs_steering captures sustained steering intensity
- steering_range captures broad directional sweep

## 10. Iterative Tuning History (Summary)

Main refinements completed:

- Shift from signal-only heuristics to explicit window-based inference
- Added overlap handling and voting
- Added all-driver mega-graphs for analysis
- Refactored code into clear functional components with documented configs
- For overtaking:
  - moved through fuzzy scoring, then stateful rules, then start-focused strict logic
  - corrected steering sign convention (left/right inversion fix)
  - forced non-overlapping previous-window comparison
- For U-Turnings:
  - introduced steering-heavy rule
  - increased strictness thresholds in two stages

## 11. Deliverables Produced

Artifacts currently generated by running stopping.py:

- stopping_megagraph.png
- overtaking_megagraph.png
- u_turnings_megagraph.png

Each mega-graph includes one subplot per driver and overlays:

- Predicted maneuver
- Marker flag maneuver

## 12. Strengths and Limitations

### 12.1 Strengths

- Fully interpretable decision logic
- Explicit fuzzy-style rules and thresholds
- Maneuver-specific customization on a shared robust pipeline
- Reproducible outputs across all five drivers

### 12.2 Limitations

- Marker labels are imperfect and may not align exactly at boundaries
- Overtaking final behavior is rule-heavy and less purely fuzzy than stopping
- Fixed thresholds may not generalize equally to all subjects/scenarios

## 13. Alignment with Practice Intentions

The practice emphasizes understandable AI logic. This implementation satisfies that by:

- Keeping stopping detection as explicit fuzzy inference with weighted antecedents
- Exposing all thresholds and weights in configuration
- Avoiding black-box training for the baseline
- Producing visual evidence across all drivers

In particular, stopping provides the clearest fuzzy-rule reference implementation, while overtaking and U-turnings demonstrate how fuzzy and rule-based reasoning can be combined pragmatically under real label noise.

## 14. Suggested Next Steps

If further refinement is required, recommended order:

1. Quantitative metrics per maneuver and per driver (precision, recall, F1).
2. Mild per-maneuver threshold calibration from marker distributions.
3. Optional confidence score export per window for analysis.
4. Keep current rules as baseline and compare against a lightweight fuzzy-neural extension only as an additional experiment.

## 15. Reproducibility

Run command:

- .venv/bin/python stopping.py

Expected outputs in workspace root:

- stopping_megagraph.png
- overtaking_megagraph.png
- u_turnings_megagraph.png
