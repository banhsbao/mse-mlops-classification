# MLOps Classification Pipeline

A simple MLOps pipeline for training and tuning a neural network classifier using PyTorch and MLflow.

## Setup

1. Clone this repository
2. Make sure you have Python 3.9+ installed
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

You can run the entire pipeline using the provided shell script:

```bash
./run_pipeline.sh
```

This script will:
1. Initialize MLflow
2. Generate synthetic classification data
3. Train a base model
4. Tune the model hyperparameters

## Running Individual Steps

You can also run individual steps using MLflow:

```bash
# Initialize MLflow
mlflow run . -e init_mlflow

# Generate synthetic data
mlflow run . -e generate_data

# Train a base model
mlflow run . -e train

# Tune hyperparameters
mlflow run . -e tune

# Run the web app
mlflow run . -e app
```

## Viewing MLflow UI

To view experiment results and metrics in the MLflow UI, use the provided script:

```bash
# Start MLflow UI on default port 5000
./mlflow_ui.sh

# Start MLflow UI on custom port
./mlflow_ui.sh -p 5002

# Stop all running MLflow UI servers
./mlflow_ui.sh stop
```

## Project Structure

- `src/` - Source code
  - `data_utils.py` - Functions for generating and loading data
  - `model.py` - Neural network model definition
  - `train.py` - Training script
  - `tune.py` - Hyperparameter tuning script
  - `init_mlflow.py` - MLflow initialization script
- `app/` - Web application code
- `data/` - Stored datasets
- `models/` - Saved model files
- `mlruns/` - MLflow experiment tracking data
- `MLproject` - MLflow project definition
- `conda.yaml` - Conda environment definition

## License

This project is open source under the MIT license. 