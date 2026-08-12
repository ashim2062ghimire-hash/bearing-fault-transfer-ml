import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

FEATURE_NAMES = [
    "RMS", "Kurtosis", "Peak_Amplitude", "Standard_Deviation",
    "Dominant_Frequency", "Peak_FFT_Amplitude", "BPFO_Amplitude",
    "BPFI_Amplitude", "BPFO_Ratio", "BPFI_Ratio"
]

def compute_feature_shift(source_X, target_X, feature_names):
    shifts = {}
    for i, name in enumerate(feature_names):
        distance = wasserstein_distance(source_X[:, i], target_X[:, i])
        shifts[name] = round(distance, 4)
    return shifts

def shift_main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    X_cwru = np.load("data/splits/X_cwru.npy")
    
    target_domains = ["MFS", "self_built"]
    
    for domain in target_domains:
        X_target = np.load(f"data/splits/X_{domain}.npy")
        
        shift_dict = compute_feature_shift(X_cwru, X_target, FEATURE_NAMES[:X_cwru.shape[1]])
        
        df_shift = pd.DataFrame(list(shift_dict.items()), columns=["Feature", "Wasserstein_Distance"])
        df_shift["Shift_Rank"] = df_shift["Wasserstein_Distance"].rank(ascending=True, method="min").astype(int)
        df_shift = df_shift.sort_values("Wasserstein_Distance", ascending=True)
        
        output_csv = os.path.join(output_dir, f"{domain}_feature_shift_rank.csv")
        df_shift.to_csv(output_csv, index=False)
        
        print(f"--> Saved feature shift ranking for {domain} to '{output_csv}'")

if __name__ == "__main__":
    shift_main()