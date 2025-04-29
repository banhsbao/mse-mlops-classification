# Kiến trúc mô hình Neural Network

## Tổng quan

Dự án sử dụng một mô hình Multi-Layer Perceptron (MLP) được xây dựng bằng PyTorch để giải quyết các bài toán phân loại. Mô hình được thiết kế với độ linh hoạt cao, cho phép điều chỉnh số lượng lớp ẩn, số lượng neuron trên mỗi lớp và tỷ lệ dropout để phù hợp với nhiều loại dữ liệu khác nhau.

## Lớp MLPClassifier

```python
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=2, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
```

## Các thành phần chính

1. **Input Layer**:
   - Nhận đầu vào với kích thước được chỉ định bởi `input_dim`
   - Thường tương ứng với số lượng features trong dữ liệu

2. **Hidden Layers**:
   - Số lượng và kích thước được xác định bởi tham số `hidden_dims`
   - Mặc định là hai lớp ẩn với kích thước [64, 32]
   - Sử dụng hàm kích hoạt ReLU sau mỗi lớp tuyến tính

3. **Dropout Regularization**:
   - Tỷ lệ dropout được kiểm soát bởi tham số `dropout_rate`
   - Mặc định là 0.2 (20% neuron bị vô hiệu hóa trong quá trình huấn luyện)
   - Giúp ngăn ngừa overfitting

4. **Output Layer**:
   - Kích thước xác định bởi `output_dim`
   - Thường là số lượng classes trong bài toán phân loại

## Phương thức forward

```python
def forward(self, x):
    return self.model(x)
```

Phương thức `forward` đơn giản chuyển đầu vào qua toàn bộ mô hình tuần tự đã định nghĩa.

## Phương thức dự đoán

```python
def predict(self, x):
    with torch.no_grad():
        logits = self(x)
        _, predicted = torch.max(logits, 1)
        return predicted
```

Phương thức `predict` trả về class được dự đoán với xác suất cao nhất.

## Phương thức tính xác suất

```python
def predict_proba(self, x):
    with torch.no_grad():
        logits = self(x)
        return F.softmax(logits, dim=1)
```

Phương thức `predict_proba` trả về xác suất thuộc về mỗi class bằng cách áp dụng hàm softmax lên logits.

## Tối ưu hóa hyperparameters

Các hyperparameters của mô hình có thể được tối ưu hóa, bao gồm:

- Số lượng hidden layers
- Số lượng neurons trong mỗi layer
- Tỷ lệ dropout
- Learning rate
- Batch size
- Số lượng epochs

Quá trình tối ưu được thực hiện thông qua `hyperopt` và được theo dõi bằng MLflow. 