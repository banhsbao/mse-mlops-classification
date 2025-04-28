# MLOps Pipeline with PyTorch, MLflow and Flask

This project demonstrates a complete MLOps pipeline that includes:
- Data generation using sklearn's make_classification
- PyTorch model implementation (MLP classifier)
- Training and evaluation
- Hyperparameter tuning
- Experiment tracking with MLflow
- Model registry
- Flask web application for serving predictions

## Project Structure

```
.
├── app/                      # Flask web application
│   ├── app.py                # Main Flask application
│   └── templates/            # HTML templates
│       ├── index.html        # Input form
│       └── result.html       # Prediction results
├── data/                     # Data directory (created automatically)
├── src/                      # Source code
│   ├── data_utils.py         # Data generation utilities
│   ├── model.py              # PyTorch MLP model
│   ├── train.py              # Training script
│   └── tune.py               # Hyperparameter tuning
├── conda.yaml                # Conda environment specification
├── python_env.yaml           # Python environment for MLflow
├── MLproject                 # MLflow project definition
├── predict_client.py         # Test client for Flask API
├── requirements.txt          # Python dependencies
└── run.py                    # Script to run the pipeline
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Option 1: Run each step individually

You can run each step of the pipeline manually:

```bash
python run.py --step generate

python run.py --step train

python run.py --step tune

python run.py --step app
```

### Option 2: Run all steps at once

```bash
python run.py --step all
```

### Option 3: Run using MLflow

```bash
mlflow run . -e generate_data

mlflow run . -e train

mlflow run . -e tune

mlflow run . -e app
```

## Using the Web Application

1. Start the Flask application:
   ```
   python run.py --step app
   ```

2. Open your web browser and navigate to http://localhost:5000

3. Enter 20 comma-separated feature values in the form and click "Predict"

4. View the prediction result

## Testing the API

You can test the API directly using the provided client:

```bash
python predict_client.py

python predict_client.py --random

python predict_client.py --features "0.5,1.2,-0.3,0.8,1.5,-0.7,0.2,0.9,-1.1,0.4,0.6,-0.5,1.0,-0.2,0.3,0.7,-0.9,1.3,-0.4,0.1"

python predict_client.py --generate
```

## Viewing Experiment Results

To view the MLflow tracking UI:

```bash
mlflow ui --port 5001
```

Then open your browser and navigate to http://localhost:5001 to explore your experiments. 