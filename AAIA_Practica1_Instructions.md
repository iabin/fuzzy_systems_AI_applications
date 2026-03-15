# AAIA Practice 1: What Needs To Be Done

## Goal
Design, implement, and evaluate a **fuzzy inference system** to detect driving maneuvers from simulator data.

You must detect the maneuver being performed at each instant or, more realistically, at each **time window**.

## Required Deliverable
Build a system that:
- uses **fuzzy logic** (mandatory)
- uses **sliding time windows** (mandatory)
- detects **at least 3** of these 5 maneuvers:
  - Overtaking
  - Stopping
  - Turning
  - 3-point turn
  - U-turn
- evaluates predictions against the provided labels (`Maneuver marker flag`)

## Data
Use the dataset from Aula Global:
- `ManiobrasSimulador.zip`

Dataset facts:
- 5 drivers
- 1 file per maneuver per driver
- data sampled at **20 Hz** (20 times per second)
- label column: **`Maneuver marker flag`**
  - `1` = the maneuver is happening
  - `0` = the maneuver is not happening

Useful signals:
- Speed
- RPM
- Steering Wheel angle
- Gas Pedal
- Brake Pedal
- Clutch Pedal
- Gear

## Important Constraints
- **Fuzzy systems are mandatory.**
- **Sliding windows are mandatory.**
- There are **"no maneuver"** periods in the files.
- A maneuver is **distributed over time**, not a single sample.
- You must **analyze overlapping vs non-overlapping windows**.
- You must **analyze different window sizes**.
- `Left turn` and `Right turn` **cannot be used** to recognize turn maneuvers.
- If `gear = 0` between two gear values, treat it as a **gear transition**, not a stable gear.

## Work Plan

### 1. Explore and preprocess the data
- Load the Excel files.
- Inspect columns and example rows.
- Check for missing values, errors, or outliers.
- Decide how to handle them and justify it.
- Optionally normalize/scale signals if it helps define membership functions.
- Separate data for:
  - system design/calibration
  - final evaluation

### 2. Build temporal windows
- Define a window size in samples or seconds.
- Use sliding windows.
- Try both:
  - overlapping windows
  - non-overlapping windows
- Compare which works better.

### 3. Extract features per window
Possible features:
- mean / max / min speed
- steering angle variation
- average brake usage
- average accelerator usage
- number of gear changes
- other window-level features that help distinguish maneuvers

These features will be the inputs to the fuzzy system.

### 4. Define fuzzy variables
For each selected maneuver, choose input variables that make sense.

Examples:
- **Overtaking**
  - current speed
  - speed variation
  - possible lane-change behavior inferred from steering angle evolution
- **Stopping**
  - average speed in the window
  - brake usage level
- **Turns / U-turn / 3-point turn**
  - steering angle
  - steering variation
  - speed reduction
  - gear changes

Then define:
- fuzzy input variables
- fuzzy output variables
- membership functions

### 5. Create the fuzzy rule base
Write IF-THEN rules based on domain knowledge and what you observe in the data.

Example ideas:
- IF speed is high AND steering changes slightly AND accelerator is pressed, THEN overtaking is high
- IF speed is low AND brake is strong, THEN stopping is high
- IF speed is low AND steering angle is large, THEN turning is high

### 6. Implement the fuzzy system
You can do either:
- one fuzzy system per maneuver, or
- one multi-output fuzzy system

Also choose:
- inference method
- defuzzification method

Example suggested in the PDF:
- Mamdani + centroid

Apply the system to all windows.

### 7. Decide the predicted maneuver
For each window:
- compute fuzzy degree for each maneuver
- choose the maneuver with the highest degree
- only assign it if it passes a threshold
- otherwise classify as **no maneuver**

## Evaluation
Use `Maneuver marker flag` as reference.

You should evaluate whether:
- when `flag = 1`, your system gives a high degree for that maneuver
- when `flag = 0`, your system gives a low degree

Suggested metrics:
- Accuracy
- Sensitivity / TPR
- Specificity or false positive rate

Also include temporal plots:
- compare `Maneuver marker flag` vs fuzzy output over time

## Analysis You Must Include
Discuss:
- which maneuvers work best
- which maneuvers work worst
- whether overlapping windows help
- whether window size matters
- which rules or membership functions could be improved

## Submission Notes
- Work in groups of **3 or 4**
- Only **one group member** submits
- Submission is only for **feedback**
- Final evaluation is an **individual exam** on **18 March 2026**
- The PDF says delivery is due in Aula Global before **23 February 2025 at 15:00**, which looks inconsistent with the 2025/26 course and is likely a typo

## Minimal Pipeline

```text
Load data
→ clean / preprocess
→ segment into sliding windows
→ extract features
→ define fuzzy variables + membership functions
→ write rules
→ run fuzzy inference
→ assign maneuver / no maneuver
→ evaluate against Maneuver marker flag
→ analyze results
```
