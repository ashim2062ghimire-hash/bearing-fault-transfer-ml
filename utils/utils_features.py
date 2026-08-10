import numpy as np
import os
import json
import random
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import joblib

def set_reproducibility(seed=9):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"-> Reproducibility seed established globally at: {seed}")

def bearing_fault_frequencies(rpm):
    d = (5/16) 
    D = 1.537
    n = 9
    theta = 0.0
    fr = rpm / 60.0
    ratio = (d / D) * np.cos(theta)

    bpfo = (n / 2) * fr * (1 - ratio)
    bpfi = (n / 2) * fr * (1 + ratio)
    return bpfo, bpfi

def segment_signal(signal, seg_len, overlap):
    step = int(seg_len * (1 - overlap))
    segments = [
        signal[i:i + seg_len]
        for i in range(0, len(signal) - seg_len + 1, step)
    ]
    return np.array(segments)

def label_dataset(file_path, mapping_path='data/mapping/fault_group_mapping.json'):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    if base_name in mapping:
        return mapping[base_name]
    raise KeyError(f"The file named {base_name} cannot be located.")

def normalize_dataset(prefix):
    input_csv = f'data/processed/{prefix}_features_comp.csv'
    output_csv = f'data/processed/{prefix}_features_norm.csv'
    scaler_path = "data/processed/cwru_scaler.pkl"
 
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The file '{input_csv}' does not exist!")
        
    df = pd.read_csv(input_csv)
    
    feature_cols = df.select_dtypes(include=[np.number]).columns[:-2]
    if prefix=="cwru":
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, scaler_path)
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file '{scaler_path}' not found! Process 'cwru' first.")
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols])
    df.to_csv(output_csv, index=False, header=False)
    print(f" -> Successfully saved normalized {prefix} dataset.")

def feature_calc(rpm, final_signal, prefix, final_fs):
    bpfo, bpfi = bearing_fault_frequencies(rpm)
    segment = segment_signal(signal = final_signal, seg_len = 2000, overlap = 0.5)
    features = []
    print(f"{len(segment)} segments for {prefix}")
    for segment_i in segment:
        n = len(segment_i)
        rms = np.sqrt(np.mean(segment_i ** 2))
        kurt = scipy.stats.kurtosis(segment_i, fisher=True)
        peak_amp = np.max(np.abs(segment_i))
        std_dev = np.std(segment_i)

        window = np.hanning(n)
        fft_vals = np.abs(np.fft.rfft(segment_i * window))
        freqs = np.fft.rfftfreq(n, d=1 / final_fs)
        dominant_freq = freqs[np.argmax(fft_vals)]
        peak_fft_amp = np.max(fft_vals)

        analytic = scipy.signal.hilbert(segment_i)
        envelope = np.abs(analytic)
        env_fft = np.abs(np.fft.rfft((envelope - np.mean(envelope)) * window))
        env_freqs = np.fft.rfftfreq(n, d=1 / final_fs)
        def sum_amp_near(target_freq, multipliers=[1, 2, 3]):
            sum = 0.0
            for m in multipliers:
                mask = np.abs(env_freqs - m*target_freq) <= 5.0
                if np.any(mask):
                    sum += env_fft[mask].max()
            return sum
        bpfo_amp = sum_amp_near(bpfo)
        bpfi_amp = sum_amp_near(bpfi)
        amp_sum = bpfi_amp + bpfo_amp
        bpfo_ratio = bpfo_amp / amp_sum
        bpfi_ratio = bpfi_amp / amp_sum
        features.append([ rms, kurt, peak_amp, std_dev, dominant_freq, peak_fft_amp,
            bpfo_amp, bpfi_amp, bpfo_ratio, bpfi_ratio])
    return features