import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.manifold import TSNE
from utils.utils_ml_evaluate import load_class_mapping, load_X_y_groups

FEATURE_NAMES = [
            "RMS", "Kurtosis", "Peak_Amplitude", "Standard_Deviation", "Dominant_Frequency", "Peak_FFT_Amplitude", "BPFO_Amplitude", 
            "BPFI_Amplitude", "BPFO_Ratio", "BPFI_Ratio"
        ]

CLASS_NAMES, class_labels = load_class_mapping()

def beeswarm_plots(model, X_data, domain_name, output_dir, target_class_idx=6):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    X_df = pd.DataFrame(X_data, columns=FEATURE_NAMES)
    os.makedirs(output_dir, exist_ok=True)

    for idx, class_name in enumerate(CLASS_NAMES):
        plt.figure(figsize=(10, 6))

        target_shap = shap_values[:, :, idx] 

        shap.summary_plot(
            target_shap, 
            X_df, 
            feature_names=FEATURE_NAMES,
            plot_type="dot", 
            show=False
        )
        
        plt.title(f"{domain_name.upper()} - Directional Beeswarm ({class_name})", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"beeswarm/{domain_name}_beeswarm_{class_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    print(f"--> Saved SINGLE Beeswarm plot for {domain_name}.")


def plot_tsne_cluster_overlap(output_dir, domain):
    X_target, y_target, groups_target= load_X_y_groups(prefix=domain)
    X_cwru, y_cwru, groups_cwru= load_X_y_groups(prefix='cwru')
    combined_features = np.vstack([X_target, X_cwru])
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=9, n_jobs=-1)
    X_2d = tsne.fit_transform(combined_features)
    
    X_target_2d = X_2d[:len(X_target)]
    X_cwru_2d = X_2d[len(X_target):]
    
    plt.figure(figsize=(12, 9))

    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    
    for c in CLASS_NAMES:
        class_name = class_labels[c]
        color = colors[c]
        
        mask_target = (y_target == c)
        mask_cwru = (y_cwru == c)
        
        if np.any(mask_target):
            plt.scatter(
                X_target_2d[mask_target, 0], X_target_2d[mask_target, 1], 
                facecolors='none', edgecolors=color, linewidths=1.5,
                alpha=0.6, s=25, label=f"{domain} - {class_name}"
            )
            
        if np.any(mask_cwru):
            plt.scatter(
                X_cwru_2d[mask_cwru, 0], X_cwru_2d[mask_cwru, 1], 
                color=color, alpha=0.6, s=25, label=f"CWRU - {class_name}"
            )
            
    plt.title(f"t-SNE Space Alignment: {domain} (Hollow) vs CWRU (Solid)", fontsize=14, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    tsne_path = os.path.join(output_dir, f"t-SNE/{domain}_tsne_plot.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    print(f"--> Saved t-SNE plot for {domain}.")

def plot_main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load("model/random_forest_model.joblib")

    domains = ["MFS", "self_built"]
    domains_data = {}

    for domain in domains:
        X = np.load(f"data/splits/X_{domain}.npy")

        beeswarm_plots(model, X, domain, output_dir)

        plot_tsne_cluster_overlap(output_dir, domain)


if __name__ == "__main__":
    plot_main()