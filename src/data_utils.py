import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle


def generate_data(n_samples=1000, n_features=20, n_classes=2, random_state=42, test_size=0.2):
    """Generate synthetic classification data and split into train/test sets"""
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'n_features': n_features,
        'n_classes': n_classes
    }
    
    os.makedirs('data', exist_ok=True)
    
    with open('data/classification_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data generated and saved to data/classification_data.pkl")
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return data


def load_data():
    """Load the generated classification data"""
    try:
        with open('data/classification_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("Data file not found. Generating new data...")
        return generate_data()


if __name__ == "__main__":
    generate_data() 