from src.features_extract.step01_cwru_extraction import cwru_main
from src.features_extract.step02_valid_extraction import valid_main
from src.ml.step03_preprocessing import preprocess_pipeline
from src.ml.step04_training_and_evaluation import train_and_evaluate_simulation
from src.evaluate.step05_domain_evaluate import evaluate_pipeline
from src.evaluate.step06_feature_shift import shift_main
from src.evaluate.step07_tSNE_SHAP import plot_main

def main():
     print("|| STARTING FEATURE EXTRACTION PIPELINE ||")
     cwru_main()
     valid_main()

     print("|| STARTING ML PIPELINE ||")
     preprocess_pipeline(
          output_dir="data/splits"
     )
     train_and_evaluate_simulation(
          output_dir="output", 
          model_dir="model"
     )

     print("|| STARTING EVALUATION PIPELINE ||")
     evaluate_pipeline(
          model_dir="model/random_forest_model.joblib", 
          output_dir="output"
     )
     shift_main()
     plot_main()

     print("|| COMPLETE PIPELINE EXECUTED SUCCESSFULLY ||")

if __name__ == "__main__":
    main()