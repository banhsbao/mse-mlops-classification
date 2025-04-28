import argparse
import requests
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def generate_random_features(n_features=20):
    """Generate random feature values for testing"""
    return np.random.uniform(-3, 3, n_features).tolist()

def predict(features, url="http://localhost:5000/predict"):
    """Send prediction request to Flask API"""
    
    print(f"Input features: {features}")
    
    response = requests.post(
        url,
        json={"features": features},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Result:")
        print(f"Class: {result['prediction']} ({result['prediction_label']})")
        print("\nClass Probabilities:")
        for i, prob in enumerate(result['probabilities']):
            print(f"Class {i}: {prob:.4f}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def main():
    parser = argparse.ArgumentParser(description='Test the Flask ML API')
    parser.add_argument('--features', type=str, 
                        help='Comma-separated feature values (default: uses default features)')
    parser.add_argument('--random', action='store_true',
                        help='Use random feature values')
    parser.add_argument('--generate', action='store_true',
                        help='Generate data if it does not exist')
    parser.add_argument('--url', type=str, default="http://localhost:5000/predict",
                        help='URL of the prediction API')
    
    args = parser.parse_args()
    
    if args.generate and not os.path.exists('data/classification_data.pkl'):
        print("Generating classification data...")
        from src.data_utils import generate_data
        generate_data()
    
    if args.random:
        features = generate_random_features()
    elif args.features:
        features = [float(f.strip()) for f in args.features.split(',')]
    else:
        features = [0.5, 1.2, -0.3, 0.8, 1.5, -0.7, 0.2, 0.9, -1.1, 0.4, 
                   0.6, -0.5, 1.0, -0.2, 0.3, 0.7, -0.9, 1.3, -0.4, 0.1]
    
    predict(features, args.url)

if __name__ == "__main__":
    main() 