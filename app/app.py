import os
import sys
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
import mlflow.pytorch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.model import MLPClassifier
from src.data_utils import load_data

app = Flask(__name__)

def load_model():
    try:
        model = mlflow.pytorch.load_model("models:/MLPClassifier/latest")
        print("Loaded registered model from MLflow Model Registry")
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        print("Falling back to locally saved model...")
        
        try:
            data = load_data()
            n_features = data['n_features']
            n_classes = data['n_classes']
            
            model = MLPClassifier(input_dim=n_features, output_dim=n_classes)
            
            model.load_state_dict(torch.load("../models/mlp_classifier.pt"))
            model.eval()
            print("Loaded model from local file")
        except Exception as e2:
            print(f"Error loading local model: {e2}")
            print("Using a default model...")
            
            model = MLPClassifier(input_dim=20, output_dim=2)
            model.eval()
            print("Using default model")
            
    return model

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type == 'application/json':
            data = request.get_json()
            features = data.get('features', [])
        else:
            features = request.form.get('features', '')
            if isinstance(features, str):
                features = [float(x.strip()) for x in features.split(',') if x.strip()]
        
        features_tensor = torch.FloatTensor([features])
        
        with torch.no_grad():
            prediction = model.predict(features_tensor).item()
            probas = model.predict_proba(features_tensor).squeeze().tolist()
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Class ' + str(prediction),
            'probabilities': probas
        }
        
        if request.content_type == 'application/json':
            return jsonify(result)
        else:
            prob_widths = [f"{int(p * 100)}%" for p in probas]
            return render_template('result.html', 
                                  prediction=result['prediction_label'],
                                  probabilities=result['probabilities'],
                                  prob_widths=prob_widths,
                                  features=features)
            
    except Exception as e:
        error_msg = str(e)
        if request.content_type == 'application/json':
            return jsonify({'error': error_msg}), 400
        else:
            return render_template('index.html', error=error_msg)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=1111) 