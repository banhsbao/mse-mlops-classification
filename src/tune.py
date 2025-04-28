import os
import sys
import mlflow
import mlflow.pytorch
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train


def tune():
    """Tune hyperparameters and track with MLflow"""
    mlflow.set_experiment("mlp_hyperparameter_tuning")
    
    param_grid = {
        'hidden_dims': [[32, 16], [64, 32], [128, 64], [64, 32, 16]],
        'lr': [0.01, 0.001, 0.0001],
        'batch_size': [16, 32, 64],
        'epochs': [15, 20, 25],
        'dropout_rate': [0.1, 0.2, 0.3]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = [
        {'hidden_dims': [64, 32], 'lr': 0.001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [32, 16], 'lr': 0.001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [128, 64], 'lr': 0.001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [64, 32, 16], 'lr': 0.001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [64, 32], 'lr': 0.01, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [64, 32], 'lr': 0.0001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [64, 32], 'lr': 0.001, 'batch_size': 16, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [64, 32], 'lr': 0.001, 'batch_size': 64, 'epochs': 20, 'dropout_rate': 0.2},
        {'hidden_dims': [64, 32], 'lr': 0.001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.1},
        {'hidden_dims': [64, 32], 'lr': 0.001, 'batch_size': 32, 'epochs': 20, 'dropout_rate': 0.3},
    ]
    
    best_accuracy = 0
    best_model = None
    best_params = None
    best_run_id = None
    
    print(f"Running {len(combinations)} hyperparameter combinations...")
    
    with mlflow.start_run(run_name="hyperparameter_tuning_parent") as parent_run:
        for idx, params in enumerate(combinations):
            print(f"\nTraining model {idx+1}/{len(combinations)}")
            print(f"Parameters: {params}")
            
            with mlflow.start_run(nested=True, run_name=f"model_{idx}") as run:
                model, accuracy = train(**params)
                
                run_id = run.info.run_id
                mlflow.set_tag("model_index", idx)
                
                print(f"Run ID: {run_id}, Accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = params
                    best_run_id = run_id
                    print(f"New best model found!")
        
        print("\n=== Best Model ===")
        print(f"Run ID: {best_run_id}")
        print(f"Parameters: {best_params}")
        print(f"Accuracy: {best_accuracy:.4f}")
        
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_param("best_params", best_params)
        mlflow.log_metric("best_accuracy", best_accuracy)
        
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, "MLPClassifier")
        
        print(f"Model registered as: MLPClassifier, version: {mv.version}")
    
    return best_model, best_params, best_accuracy


if __name__ == "__main__":
    tune() 