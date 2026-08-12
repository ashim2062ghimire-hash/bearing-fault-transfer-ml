import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

def load_class_mapping(mapping_path="data/mapping/class_mapping.json"):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Missing required class map file at: '{mapping_path}'")
        
    with open(mapping_path, "r") as f:
        fault_classes = json.load(f)
        
    unique_classes = np.array(sorted([int(k) for k in fault_classes.keys()]))
    
    display_labels = [fault_classes[str(k)] for k in unique_classes]
    
    return unique_classes, display_labels

def load_X_y_groups(prefix):
    X_path = f"data/splits/X_{prefix}.npy"
    y_path = f"data/splits/y_{prefix}.npy"
    groups_path = f"data/splits/groups_{prefix}.npy"
    
    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(groups_path)):
        raise FileNotFoundError(f"Missing required split files for prefix: '{prefix}'")
        
    X = np.load(X_path)
    y = np.load(y_path)
    groups = np.load(groups_path)
    
    return X, y, groups

def cm_f1(y_true, y_pred, output_dir, prefix, unique_classes, display_labels):
    f1 = f1_score(y_true, y_pred, average='macro')
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{prefix}_f1.npy"), f1)
    print(f"-> Successfully saved f1-score to 'output/{prefix}_f1.npy'")
        
    print("Generating visual Confusion Matrix plot...")
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_classes, normalize='true')
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp_cm.plot(cmap=plt.cm.Oranges, ax=ax, values_format='.2f', xticks_rotation=45)
    plt.title(f"{prefix.upper()} Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Actual Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(f'output/{prefix}_confusion_matrix.png', dpi=300)
    plt.close()
    print(f"-> Successfully saved CM plot to 'output/{prefix}_confusion_matrix.png'")

def collapse_to_3class(y_7class):
    mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
    return np.array([mapping[val] for val in y_7class])

def compute_error_breakdown(y_true, y_pred):
    y_true_3c = collapse_to_3class(y_true)
    y_pred_3c = collapse_to_3class(y_pred)
    total = len(y_true)
    
    loc_errors = np.sum(y_true_3c != y_pred_3c)
    sev_errors = np.sum((y_true_3c == y_pred_3c) & (y_true != y_pred))
    
    return {
        "Location Error Rate (%)": round((loc_errors / total) * 100, 2),
        "Severity Error Rate (%)": round((sev_errors / total) * 100, 2)
    }