#!/bin/bash

echo "Initializing MLflow..."
mlflow run . -e init_mlflow

echo "Generating data..."
mlflow run . -e generate_data

echo "Training model..."
mlflow run . -e train

echo "Tuning hyperparameters..."
mlflow run . -e tune

echo "Pipeline completed successfully!" 