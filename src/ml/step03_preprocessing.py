import os
import pandas as pd
import numpy as np

def preprocess_pipeline(output_dir):

    print(f"\n || Starting Preprocessing Pipeline || \n")

    x = ["cwru", "MFS", "self_built"]
    for i in x:
        csv = f"data/processed/{i}_features_norm.csv"
    
        if not (os.path.exists(f"data/processed/{i}_features_norm.csv")):
            raise FileNotFoundError(f"Processed {i} feature file not found. Please run feature extraction first.")
            
        df = pd.read_csv(csv) 

        print(f"Loaded cwru dataset with shape: {df.shape} (Rows x Columns)")
        X = df.iloc[:,:-2].values
        y = df.iloc[:,-1].values
        groups = df.iloc[:,-2].values

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"X_{i}.npy"), X)
        np.save(os.path.join(output_dir, f"groups_{i}.npy"), groups)
        np.save(os.path.join(output_dir, f"y_{i}.npy"), y)
        
        print(f"Isolated {X.shape[1]} engineering features for {i}.")
        print("---Saved splitted cwru files to 'data/splits/' ---")

    print("---Pre-processing completed---")
    
if __name__ == "__main__":
    preprocess_pipeline()