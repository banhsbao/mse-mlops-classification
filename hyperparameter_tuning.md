# Tối ưu hóa Hyperparameters

## Tổng quan

Quá trình tối ưu hóa hyperparameters là một bước quan trọng để cải thiện hiệu suất của mô hình phân loại. Dự án sử dụng thư viện Hyperopt kết hợp với MLflow để tìm kiếm cấu hình tham số tối ưu cho mô hình neural network.

## Các hyperparameters được tối ưu hóa

1. **Kiến trúc mạng**
   - Số lượng hidden layers
   - Số lượng neurons trong mỗi layer

2. **Hyperparameters huấn luyện**
   - Learning rate
   - Batch size
   - Số lượng epochs

3. **Regularization**
   - Tỷ lệ dropout

## Không gian tham số (Search Space)

Không gian tìm kiếm được định nghĩa như sau:

```python
search_space = {
    'hidden_dims': hp.choice('hidden_dims', [
        [32],
        [64],
        [128],
        [64, 32],
        [128, 64],
        [256, 128],
        [128, 64, 32],
        [256, 128, 64]
    ]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'epochs': hp.choice('epochs', [30, 50, 100]),
}
```

Không gian này bao gồm:
- 8 cấu hình kiến trúc mạng khác nhau
- Learning rate từ 0.0001 đến 0.01 theo phân phối log-uniform
- 4 lựa chọn batch size
- Tỷ lệ dropout từ 0.1 đến 0.5 theo phân phối đều
- 3 lựa chọn số lượng epochs

## Thuật toán tối ưu

Dự án sử dụng Tree of Parzen Estimators (TPE) từ Hyperopt để tìm kiếm hyperparameters tối ưu. TPE là một thuật toán hiệu quả cho không gian tham số phức tạp và có thể xử lý cả tham số liên tục và rời rạc.

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials
)
```

## Hàm mục tiêu (Objective Function)

Hàm mục tiêu huấn luyện mô hình với một bộ hyperparameters cụ thể và đánh giá hiệu suất:

```python
def objective(params):
    with mlflow.start_run(nested=True) as run:
        # Extract params
        hidden_dims = params['hidden_dims']
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        dropout_rate = params['dropout_rate']
        epochs = params['epochs']
        
        # Log params to MLflow
        mlflow.log_params(params)
        
        # Load data
        data = load_data()
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        n_features = data['n_features']
        n_classes = data['n_classes']
        
        # Create and train model
        model = MLPClassifier(
            input_dim=n_features,
            hidden_dims=hidden_dims,
            output_dim=n_classes,
            dropout_rate=dropout_rate
        )
        
        # Training logic...
        
        # Evaluate on test set
        test_accuracy, test_loss, test_f1 = evaluate_model(model, X_test, y_test)
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'test_f1': test_f1
        })
        
        # We want to maximize accuracy, so return negative accuracy as loss
        return {'loss': -test_accuracy, 'status': STATUS_OK}
```

## Tracking và so sánh kết quả

MLflow theo dõi mỗi lần thực hiện hàm mục tiêu như một run riêng biệt, cho phép so sánh các cấu hình khác nhau. Để phân tích kết quả, MLflow UI cung cấp các công cụ trực quan:

1. **Parallel Coordinates Plot** - Biểu diễn trực quan mối quan hệ giữa các hyperparameters và metrics
2. **Scatter Plots** - So sánh hiệu suất của các cấu hình khác nhau
3. **Parameter Importance** - Xác định các hyperparameters có ảnh hưởng lớn nhất đến hiệu suất

## Lựa chọn mô hình tốt nhất

Sau khi hoàn thành quá trình tối ưu, mô hình tốt nhất được lựa chọn dựa trên độ chính xác trên tập test. Mô hình này được lưu lại để sử dụng trong ứng dụng:

```python
# Get the best trial and its run ID
best_trial = sorted(trials.trials, key=lambda x: x['result']['loss'])[0]
best_run_id = best_trial['misc']['tid']

# Log the best model
mlflow.log_metric("best_accuracy", -best_trial['result']['loss'])
print(f"Best hyperparameters: {best_params}")
print(f"Best accuracy: {-best_trial['result']['loss']:.4f}")
```

## So sánh với mô hình cơ sở

Kết quả của mô hình tốt nhất được so sánh với mô hình cơ sở để đánh giá mức độ cải thiện:

```python
# Load the baseline model results
baseline_metrics = get_baseline_metrics()

# Calculate improvement
accuracy_improvement = (-best_trial['result']['loss'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100

print(f"Improvement over baseline: {accuracy_improvement:.2f}%")
``` 