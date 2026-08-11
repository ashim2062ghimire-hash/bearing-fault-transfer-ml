import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from utils.utils_ml import load_X_y_groups, load_class_mapping, cm_f1

def train_and_evaluate_simulation(output_dir, model_dir):
    print("\n--- [Starting RF Training with CWRU dataset] ---")
    
    X_cwru, y_cwru, groups_cwru= load_X_y_groups(prefix='cwru')
    unique_classes, display_labels= load_class_mapping()

    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=100, 
        min_samples_split=7, 
        max_features=5,
        random_state=9, 
        n_jobs=-1
    )
    sgkf = StratifiedGroupKFold(n_splits=len(np.unique(groups_cwru)))
    
    print(f"Running {len(np.unique(groups_cwru))}-Folds via Stratified Group K-Fold...")
    
    y_pred = cross_val_predict(model, X_cwru, y_cwru, groups=groups_cwru, cv=sgkf)
    
    cm_f1(y_true=y_cwru, 
          y_pred=y_pred, 
          output_dir=output_dir, 
          prefix='cwru', 
          unique_classes=unique_classes, 
          display_labels=display_labels
          )
    
    model.fit(X_cwru, y_cwru)
    
    feature_names = ["RMS", "Kurtosis", "Peak_Amplitude", "Standard_Deviation", "Dominant_Frequency", "Peak_FFT_Amplitude", 
                     "BPFO_Amplitude", "BPFI_Amplitude", "BPFO_Ratio", "BPFI_Ratio"]
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(df_imp.sort_values(by='Importance', ascending=False).to_string(index=False))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "random_forest_model.joblib"))
    print("-> Successfully saved final model to 'model/random_forest_model.joblib'")

if __name__ == "__main__":
    train_and_evaluate_simulation()