from features_extract.step01_cwru_extraction import cwru_main
from ml.step03_preprocessing import preprocess_pipeline
from ml.step04_training_and_evaluation import train_and_evaluate_simulation
from evaluate.step05_domain_evaluate import evaluate_pipeline
from evaluate.step06_feature_shift import run_domain_comparison

def main():
     print("|| STARTING CWRU FILE EXTRACTION ||")
     cwru_main()
     print("|| CWRU FILE EXTRACTION COMPLETED SUCCESSFULLY ||")

     print("|| STARTING ML PIPELINE ||")
     
     preprocess_pipeline(
          output_dir="data/splits"
     )

     train_and_evaluate_simulation(
          output_dir="output", 
          model_dir="model"
     )
     evaluate_pipeline(
          model_dir="model/random_forest_model.joblib", 
          output_dir="output"
     )
     run_domain_comparison(
          output_dir="output"
     )
     print("|| ML PIPELINE EXECUTED SUCCESSFULLY ||")

if __name__ == "__main__":
    main()