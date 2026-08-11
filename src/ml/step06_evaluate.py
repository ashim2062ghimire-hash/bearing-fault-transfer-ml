import os
import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib.pyplot as plt
from utils.utils_ml import load_class_mapping, cm_f1, load_X_y_groups

def evaluate_ansys_pipeline(output_dir, model_dir):
    print("\n--- [Initializing Cross-Domain Evaluation] ---")
    
    model_path = os.path.join(model_dir)

    print("Loading datasets, random forest classification model, and class maps...")

    X_ansys, y_ansys, groups_ansys= load_X_y_groups(prefix='ansys')
    model = joblib.load(model_path)
    unique_classes, display_labels= load_class_mapping()
    
    print(f"Extracted real-world target data matrix with shape: {X_ansys.shape}")
    print("Running model inference to predict fault categories and calculate class probabilities...")
    y_pred = model.predict(X_ansys)
    y_proba = model.predict_proba(X_ansys)
    
    cm_f1(y_true=y_ansys, 
          y_pred=y_pred, 
          output_dir=output_dir, 
          prefix='ansys', 
          unique_classes=unique_classes, 
          display_labels=display_labels
          )

    print("Computing multi-class Receiver Operating Characteristic (ROC) boundaries...")
    y_ansys_bin = label_binarize(y_ansys, classes=unique_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, c in enumerate(unique_classes):
        RocCurveDisplay.from_predictions(
            y_ansys_bin[:, i], y_proba[:, i], name=f"Class {display_labels[c]}", ax=ax
        )
    plt.plot([0, 1], [0, 1], "k--", label="AUC = 0.5")
    plt.title("CWRU Multi-Class ROC Curves", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ansys_roc_auc_curve.png"), dpi=300)
    plt.close()
    print("-> Successfully saved ROC-AUC curves to 'output/ansys_roc_auc_curve.png'")
    print("--- [CWRU Evaluation Stage Complete] ---\n")

if __name__ == "__main__":
    evaluate_ansys_pipeline()