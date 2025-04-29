# Theo dõi thực nghiệm với MLflow

## Tổng quan

Dự án sử dụng MLflow để theo dõi các thực nghiệm huấn luyện mô hình và tối ưu hóa hyperparameters. MLflow cung cấp một cách nhất quán để lưu trữ, so sánh và quản lý các thí nghiệm machine learning.

## Cấu trúc MLflow trong dự án

### MLproject

File `MLproject` định nghĩa các entry points cho các bước khác nhau trong pipeline:

```yaml
name: mlops-classification

conda_env: conda.yaml

entry_points:
  init_mlflow:
    command: "python src/init_mlflow.py"
    
  generate_data:
    parameters:
      n_samples: {type: int, default: 1000}
      n_features: {type: int, default: 20}
      n_classes: {type: int, default: 2}
    command: "python -c 'from src.data_utils import generate_data; generate_data({n_samples}, {n_features}, {n_classes})'"
    
  train:
    parameters:
      hidden_dims: {type: str, default: "[64, 32]"}
      lr: {type: float, default: 0.001}
      batch_size: {type: int, default: 32}
      epochs: {type: int, default: 50}
    command: "python src/train.py --hidden_dims {hidden_dims} --lr {lr} --batch_size {batch_size} --epochs {epochs}"
    
  tune:
    parameters:
      max_evals: {type: int, default: 20}
    command: "python src/tune.py --max_evals {max_evals}"
    
  app:
    command: "python app/app.py"
```

### Tracking Metrics

Trong quá trình huấn luyện, các metrics quan trọng được theo dõi bao gồm:

1. **Accuracy** - Độ chính xác trên tập huấn luyện và tập kiểm tra
2. **Loss** - Mất mát trên tập huấn luyện và tập kiểm tra
3. **F1-score** - Điểm F1 trên tập kiểm tra
4. **Confusion Matrix** - Ma trận nhầm lẫn trên tập kiểm tra

### Logging trong train.py

```python
# Log parameters
mlflow.log_param("hidden_dims", hidden_dims)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", epochs)

# Log metrics for each epoch
for epoch in range(epochs):
    # Training logic...
    
    # Log metrics
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

# Log the final model
mlflow.pytorch.log_model(model, "model")
```

### Tối ưu hóa Hyperparameters

MLflow được tích hợp với hyperopt để theo dõi quá trình tối ưu hóa hyperparameters:

```python
def objective(params):
    with mlflow.start_run(nested=True) as run:
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Train model with these parameters
        # ...
        
        # Log final metrics
        mlflow.log_metrics({
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "val_f1": val_f1
        })
        
        return {"loss": -val_accuracy, "status": STATUS_OK}

# Run hyperparameter optimization
best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals
)
```

## Xem MLflow UI

Để xem các thí nghiệm đã thực hiện:

```bash
./mlflow_ui.sh
```

Truy cập giao diện MLflow tại `http://localhost:5000`.

## Cấu trúc thư mục MLflow

```
mlruns/
├── 0/
│   ├── <run_id_1>/
│   │   ├── artifacts/
│   │   │   └── model/
│   │   ├── metrics/
│   │   ├── params/
│   │   └── meta.yaml
│   ├── <run_id_2>/
│   └── ...
└── meta.yaml
```

Mỗi thí nghiệm được lưu trữ với các artifacts, metrics, và parameters, cho phép dễ dàng so sánh và tái tạo các thí nghiệm. 