import os
import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from utils.utils_ml_evaluate import load_class_mapping, cm_f1, load_X_y_groups, collapse_to_3class, compute_error_breakdown

def evaluate_pipeline(output_dir, model_dir):
    print("\n--- [Initializing Cross-Domain Evaluation] ---")
    unique_classes, display_labels = load_class_mapping()
    model_path = os.path.join(model_dir)
    model = joblib.load(model_path)
    target_domains = ["MFS", "self_built"]
    print("Loading datasets, random forest classification model, and class maps...")
    for domain in target_domains:
        X_target, y_target, groups_target= load_X_y_groups(prefix = target_domains)
        y_pred = model.predict(X_target)

        cm_f1(
            y_true=y_target,
            y_pred=y_pred,
            output_dir=output_dir,
            prefix=f"{domain}_7class",
            unique_classes=unique_classes,
            display_labels=display_labels,
        )

        y_target_3c = collapse_to_3class(y_target)
        y_pred_3c = collapse_to_3class(y_pred)

        cm_f1(
            y_true=y_target_3c,
            y_pred=y_pred_3c,
            output_dir=output_dir,
            prefix=f"{domain}_3class",
            unique_classes=[0, 1, 2],
            display_labels=["Healthy", "Inner Race", "Outer Race"],
        )

        unique_speeds = np.unique(groups_target)
        rpm_results_matrix = []
        for speed in unique_speeds:
            idx = (groups_target == speed)
            y_t_speed = y_target[idx]
            y_p_speed = y_pred[idx]

            per_class_f1 = f1_score(
                y_t_speed, 
                y_p_speed, 
                labels=unique_classes, 
                average=None, 
                zero_division=0
            )
            rpm_results_matrix.append(per_class_f1)

        rpm_results_array = np.array(rpm_results_matrix)
        np.save(os.path.join(output_dir, f"{domain}_rpm_per_class_f1.npy"), rpm_results_array)
        errors = compute_error_breakdown(y_target, y_pred)
        print(f"\n--- {domain.upper()} Error Analysis ---")
        print(f"Location Error Rate: {errors['Location Error Rate (%)']}%")
        print(f"Severity Error Rate: {errors['Severity Error Rate (%)']}%\n")

    print("--- [Evaluation Stage Complete] ---\n")

if __name__ == "__main__":
    evaluate_pipeline()