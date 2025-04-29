import os
import sys
import mlflow
import mlflow.pytorch
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train


def tune():
    """Tune hyperparameters and track with MLflow"""
    
    param_grid = {
        'hidden_dims': [[32, 16], [64, 32], [128, 64], [64, 32, 16]],
        'lr': [0.01, 0.001, 0.0001],
        'batch_size': [16, 32, 64],
        'epochs': [15, 20, 25],
        'dropout_rate': [0.1, 0.2, 0.3]
    }
    
    combinations = [
        {'hidden_dims': [64, 32], 'lr': 0.001, 'batch_size': 32, 'epochs': 5, 'dropout_rate': 0.2},
        {'hidden_dims': [32, 16], 'lr': 0.001, 'batch_size': 32, 'epochs': 5, 'dropout_rate': 0.2},
    ]
    
    best_accuracy = 0
    best_model = None
    best_params = None
    best_run_id = None
    
    print(f"Running {len(combinations)} hyperparameter combinations...")
    
    for idx, params in enumerate(combinations):
        print(f"\nTraining model {idx+1}/{len(combinations)}")
        print(f"Parameters: {params}")
        
        model, accuracy = train(**params)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = params
            print(f"New best model found!")
    
    print("\n=== Best Model ===")
    print(f"Parameters: {best_params}")
    print(f"Accuracy: {best_accuracy:.4f}")    
    return best_model, best_params, best_accuracy


if __name__ == "__main__":
    mlflow.set_experiment("mlp_hyperparameter_tuning")
    with mlflow.start_run(run_name="hyperparameter_tuning"):
        tune() 