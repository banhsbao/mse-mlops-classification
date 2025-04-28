import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from contextlib import nullcontext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_data
from src.model import MLPClassifier


def train(hidden_dims=[64, 32], lr=0.001, batch_size=32, epochs=20, dropout_rate=0.2):
    mlflow.set_experiment("mlp_classification")
    
    data = load_data()
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    n_features = data['n_features']
    n_classes = data['n_classes']
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = MLPClassifier(
        input_dim=n_features,
        hidden_dims=hidden_dims,
        output_dim=n_classes,
        dropout_rate=dropout_rate
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    active_run = mlflow.active_run()
    
    if active_run is None:
        run_context = mlflow.start_run()
    else:
        run_context = nullcontext()
    
    with run_context:
        mlflow.log_param("hidden_dims", hidden_dims)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("dropout_rate", dropout_rate)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            
            model.eval()
            with torch.no_grad():
                y_pred = model.predict(X_test_tensor).numpy()
                test_acc = accuracy_score(y_test, y_pred)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)
        
        model.eval()
        with torch.no_grad():
            y_pred = model.predict(X_test_tensor).numpy()
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"Final Test Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
        
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/mlp_classifier.pt")
        
        mlflow.pytorch.log_model(model, "model")
        
        return model, accuracy


if __name__ == "__main__":
    train() 