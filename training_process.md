# Quy trình huấn luyện mô hình

## Tổng quan

Quy trình huấn luyện mô hình neural network được cài đặt trong file `src/train.py`. Quy trình này sử dụng PyTorch để huấn luyện một mô hình MLPClassifier trên dữ liệu phân loại đã tạo và sử dụng MLflow để theo dõi các thí nghiệm.

## Cấu trúc quy trình huấn luyện

### 1. Cấu hình tham số

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dims', type=str, default='[64, 32]', 
                        help='Hidden dimensions as a JSON string')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    return parser.parse_args()
```

### 2. Tải dữ liệu

```python
from src.data_utils import load_data

data = load_data()
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
n_features = data['n_features']
n_classes = data['n_classes']
```

### 3. Chuẩn bị DataLoader

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

### 4. Khởi tạo mô hình và optimizer

```python
from src.model import MLPClassifier
import ast

# Parse hidden dimensions from string to list
hidden_dims = ast.literal_eval(args.hidden_dims)

# Initialize model
model = MLPClassifier(
    input_dim=n_features,
    hidden_dims=hidden_dims,
    output_dim=n_classes
)

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
```

### 5. Huấn luyện mô hình

```python
import mlflow
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("hidden_dims", hidden_dims)
    mlflow.log_param("learning_rate", args.lr)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("epochs", args.epochs)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        # Training step
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record statistics
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate training metrics
        train_loss = np.mean(train_losses)
        train_accuracy = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        # Evaluation step
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("train_f1", train_f1, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Save trained model
    torch.save(model.state_dict(), "models/mlp_classifier.pt")
    
    # Log model with MLflow
    mlflow.pytorch.log_model(model, "model")
    
    print(f"Training completed. Model saved to models/mlp_classifier.pt")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
```

## Các metrics theo dõi

Trong quá trình huấn luyện, các metrics sau được theo dõi cho mỗi epoch:

1. **Train Loss**: Mất mát trên tập huấn luyện
2. **Train Accuracy**: Độ chính xác trên tập huấn luyện
3. **Train F1 Score**: Điểm F1 trên tập huấn luyện
4. **Validation Loss**: Mất mát trên tập kiểm tra
5. **Validation Accuracy**: Độ chính xác trên tập kiểm tra
6. **Validation F1 Score**: Điểm F1 trên tập kiểm tra

## Lưu trữ mô hình

Mô hình được lưu trữ theo hai cách:

1. **PyTorch state_dict**: Lưu vào `models/mlp_classifier.pt` để sử dụng trực tiếp trong ứng dụng
2. **MLflow model**: Lưu vào MLflow để theo dõi và quản lý phiên bản

## Chiến lược huấn luyện

### Optimizer và Loss

- **Optimizer**: Adam với learning rate có thể tùy chỉnh
- **Loss function**: Cross Entropy Loss phù hợp cho bài toán phân loại

### Early Stopping

Quy trình huấn luyện có thể được mở rộng để bao gồm early stopping:

```python
# Early stopping implementation
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(args.epochs):
    # Training logic...
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Save best model
        torch.save(model.state_dict(), "models/best_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
```

## Chạy quy trình huấn luyện

Quy trình huấn luyện có thể được chạy thông qua MLflow:

```bash
mlflow run . -e train -P hidden_dims="[128, 64]" -P lr=0.001 -P batch_size=64 -P epochs=100
```

Hoặc trực tiếp:

```bash
python src/train.py --hidden_dims "[128, 64]" --lr 0.001 --batch_size 64 --epochs 100
```

## Phân tích kết quả

Sau khi huấn luyện, kết quả có thể được phân tích thông qua MLflow UI:

```bash
./mlflow_ui.sh
```

Giao diện MLflow cho phép xem biểu đồ learning curves, so sánh các runs khác nhau, và xem chi tiết về mô hình đã huấn luyện. 