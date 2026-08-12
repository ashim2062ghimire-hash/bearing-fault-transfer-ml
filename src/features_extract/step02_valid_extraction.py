import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
from utils.utils_features import normalize_dataset, feature_calc, label_dataset

def preprocess_and_calc(file_path):
    mat_data = scipy.io.loadmat(file_path)
    base_name = os.path.basename(file_path)

    best_key = None
    max_length = 0

    for key, val in mat_data.items():
        if key.startswith("__"):
            continue

        if isinstance(val, np.ndarray):
            size = val.size
            if size > max_length:
                max_length = size
                best_key = key

    if best_key is None:
        raise KeyError(f"Could not find any signal array inside '{file_path}'")

    signal = mat_data[best_key].flatten()

    if "_600" in base_name:
        rpm = 600.0
    elif "_900" in base_name:
        rpm = 900.0
    elif "_1200" in base_name:
        rpm = 1200.0

    native_fs = 51200

    sos_drift = scipy.signal.butter(N=4, Wn=10, btype='high', fs=native_fs, output='sos')
    signal_cleaned = scipy.signal.sosfiltfilt(sos_drift, signal)

    final_signal = scipy.signal.resample_poly(signal_cleaned, up=15, down=64)
    final_fs = 12000

    sos_faults = scipy.signal.butter(N=4, Wn=1000, btype='high', fs=final_fs, output='sos')
    final_signal = scipy.signal.sosfilt(sos_faults, final_signal)
    prefix=f'{base_name}'
    features = feature_calc(rpm, final_signal, prefix, final_fs = 12000)
    return features
 
def valid_main():
    print("|| STARTING TARGET DOMAIN FILE EXTRACTION ||")
    x = ["MFS","self_built"]
    output_dir='data/processed'
    for i in x:
        input_dir = glob.glob(os.path.join(f"data/raw/{i}", "*"))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,f"{i}_features_comp.csv")
        all_rows = []
        for file_path in input_dir:
            features = preprocess_and_calc(file_path)
            labels = label_dataset(file_path, prefix = i)
            for feature_row in features:
                row = np.hstack((feature_row, labels))
                all_rows.append(row)
        if all_rows:
            df = pd.DataFrame(all_rows)
            feature_headers = [
                "RMS", "Kurtosis", "Peak_Amplitude", "Standard_Deviation", "Dominant_Frequency", "Peak_FFT_Amplitude", "BPFO_Amplitude", 
                "BPFI_Amplitude", "BPFO_Ratio", "BPFI_Ratio", "Fault_Class", "Horsepower"
            ]

            df.to_csv(output_path, index=False, header=feature_headers)
        normalize_dataset(prefix = i)

if __name__ == "__main__":
    valid_main()