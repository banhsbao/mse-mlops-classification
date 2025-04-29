import os
import mlflow

def init_mlflow():
    """Initialize MLflow by creating required directories and setting up experiments"""
    
    # Create mlruns directory if it doesn't exist
    os.makedirs("mlruns", exist_ok=True)
    
    # Create the experiments
    mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    
    # Create or get the experiment
    experiment_name = "mlp_classification"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f"Using existing experiment '{experiment_name}' with ID: {experiment.experiment_id}")
    
    # Create or get the tuning experiment
    tune_experiment_name = "mlp_hyperparameter_tuning"
    try:
        tune_experiment_id = mlflow.create_experiment(tune_experiment_name)
        print(f"Created new experiment '{tune_experiment_name}' with ID: {tune_experiment_id}")
    except mlflow.exceptions.MlflowException:
        tune_experiment = mlflow.get_experiment_by_name(tune_experiment_name)
        print(f"Using existing experiment '{tune_experiment_name}' with ID: {tune_experiment.experiment_id}")
    
    return True

if __name__ == "__main__":
    init_mlflow() 