from features_extract.step02_cwru_extraction import cwru_main
from src.ml.step04_preprocessing import preprocess_pipeline
from src.ml.step05_training_and_evaluation import train_and_evaluate_simulation
from ml.step06_evaluate import evaluate_ansys_pipeline
from src.ml.step07_domain_comparision import run_domain_comparison

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
     evaluate_ansys_pipeline(
          model_dir="model/random_forest_model.joblib", 
          output_dir="output"
     )
     run_domain_comparison(
          output_dir="output"
     )
     print("|| ML PIPELINE EXECUTED SUCCESSFULLY ||")

if __name__ == "__main__":
    main()