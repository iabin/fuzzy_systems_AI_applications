import pandas as pd
import numpy as np
from stopping import (
    MANEUVER_STOPPING, MANEUVER_OVERTAKING, MANEUVER_U_TURNINGS,
    build_default_inference_config, list_maneuver_files, infer_timeline_for_file
)

def calc_metrics(df):
    if "gt_maneuver" not in df.columns:
        return None
    y_true = df["gt_maneuver"]
    y_pred = df["pred_maneuver"]
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    
    acc = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    fpr = 1 - spec
    
    return {"Acc": acc, "TPR": tpr, "Spec": spec, "FPR": fpr}

def main():
    maneuvers = [MANEUVER_STOPPING, MANEUVER_OVERTAKING, MANEUVER_U_TURNINGS]
    
    # We will test the conditions requested by the rubrick:
    configs = [
        ("Base (1.0s, overlap=0.75)", build_default_inference_config(window_seconds=1.0, overlap_ratio=0.75)),
        ("No Overlap (1.0s, overlap=0)", build_default_inference_config(window_seconds=1.0, overlap_ratio=0.0)),
        ("Small Win (0.5s, overlap=0.75)", build_default_inference_config(window_seconds=0.5, overlap_ratio=0.75)),
        ("Large Win (1.5s, overlap=0.75)", build_default_inference_config(window_seconds=1.5, overlap_ratio=0.75))
    ]

    results = []

    for m in maneuvers:
        print(f"Evaluating {m}...")
        files = list_maneuver_files(m)
        if not files:
            print(f"  No files found for {m}")
            continue
            
        for c_name, cfg in configs:
            m_metrics = []
            for f in files:
                try:
                    df = infer_timeline_for_file(f, m, cfg)
                    met = calc_metrics(df)
                    if met:
                        m_metrics.append(met)
                except Exception as e:
                    print(f"  Error processing {f.name}: {e}")
            
            if m_metrics:
                avg_acc = np.mean([x["Acc"] for x in m_metrics])
                avg_tpr = np.mean([x["TPR"] for x in m_metrics])
                avg_spec = np.mean([x["Spec"] for x in m_metrics])
                avg_fpr = np.mean([x["FPR"] for x in m_metrics])
                results.append({
                    "Maneuver": m,
                    "Config": c_name,
                    "Accuracy": f"{avg_acc:.4f}",
                    "TPR/Sens": f"{avg_tpr:.4f}",
                    "Specificity": f"{avg_spec:.4f}",
                    "FPR": f"{avg_fpr:.4f}"
                })

    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS")
    print("="*80)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
