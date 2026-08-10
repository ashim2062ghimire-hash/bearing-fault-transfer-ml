import os
import glob
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
from utils.utils_features import normalize_dataset, feature_calc, label_dataset

def preprocess_and_calc(file_path):
    mat_data = scipy.io.loadmat(file_path)
    is_healthy = "H" in os.path.basename(file_path)
    de_key = None
    rpm = None
    base_name = os.path.basename(file_path)
    for k in mat_data.keys():
        if 'DE_time' in k:
            de_key = k
        if "RPM" in k:
            rpm = float(mat_data[k].ravel()[0])
    if rpm is None or rpm == 0:
        if "_0" in base_name:
            rpm = 1797.0
        elif "_1" in base_name:
            rpm = 1772.0
        elif "_2" in base_name:
            rpm = 1750.0
        elif "_3" in base_name:
            rpm = 1730.0
    signal = mat_data[de_key].flatten()

    if is_healthy:
        native_fs = 48000
    else:
        native_fs = 12000

    sos_drift = scipy.signal.butter(N=4, Wn=10, btype='high', fs=native_fs, output='sos')
    signal_cleaned = scipy.signal.sosfiltfilt(sos_drift, signal)
    final_fs=12000
    if is_healthy == True:
        final_signal = scipy.signal.resample_poly(signal_cleaned, up=1, down=4)
    else:
        final_signal = signal_cleaned
    sos_faults = scipy.signal.butter(N=4, Wn=1000, btype='high', fs=final_fs, output='sos')
    final_signal = scipy.signal.sosfilt(sos_faults, final_signal)
    prefix=f'{base_name}'
    features = feature_calc(rpm, final_signal, prefix, final_fs = 12000)
    return features
 
def cwru_main():
    output_dir='data/processed'
    input_dir = glob.glob(os.path.join('data/raw/cwru', "*"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,"cwru_features_comp.csv")
    all_rows = []
    for file_path in input_dir:
        features = preprocess_and_calc(file_path)
        labels = label_dataset(file_path)
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
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, "data/processed/cwru_scaler.pkl")
    normalize_dataset(prefix='cwru')

if __name__ == "__main__":
    cwru_main()