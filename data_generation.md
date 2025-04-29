# Tạo và xử lý dữ liệu

## Tổng quan

Quy trình tạo và xử lý dữ liệu là bước đầu tiên trong pipeline MLOps. Dự án này sử dụng dữ liệu phân loại tổng hợp được tạo từ scikit-learn, cho phép kiểm soát các tham số như số lượng mẫu, số lượng features, và số lượng classes.

## Module data_utils.py

Module `data_utils.py` cung cấp các hàm để tạo, lưu trữ và tải dữ liệu:

```python
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
```

### Tạo dữ liệu

Hàm `generate_data` tạo dữ liệu phân loại tổng hợp và chia thành tập huấn luyện và tập kiểm tra:

```python
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
```

### Tải dữ liệu

Hàm `load_data` tải dữ liệu đã tạo từ file hoặc tạo dữ liệu mới nếu file không tồn tại:

```python
def load_data():
    """Load the generated classification data"""
    try:
        with open('data/classification_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("Data file not found. Generating new data...")
        return generate_data()
```

## Các tham số dữ liệu

### Tham số chính

1. **n_samples**: Số lượng mẫu trong tập dữ liệu (mặc định: 1000)
2. **n_features**: Số lượng features cho mỗi mẫu (mặc định: 20)
3. **n_classes**: Số lượng classes trong bài toán phân loại (mặc định: 2)
4. **random_state**: Seed để đảm bảo tính tái tạo (mặc định: 42)
5. **test_size**: Tỷ lệ chia tập test (mặc định: 0.2)

### Ví dụ về dữ liệu

Với các tham số mặc định, dữ liệu được tạo có:
- 1000 mẫu với 20 features mỗi mẫu
- 2 classes (bài toán phân loại nhị phân)
- Tập huấn luyện: 800 mẫu
- Tập kiểm tra: 200 mẫu

## Lưu trữ dữ liệu

Dữ liệu được lưu trữ dưới dạng file pickle trong thư mục `data/`:

```
data/
└── classification_data.pkl
```

File pickle chứa một dictionary với các khóa:
- `X_train`: Features của tập huấn luyện
- `X_test`: Features của tập kiểm tra
- `y_train`: Labels của tập huấn luyện
- `y_test`: Labels của tập kiểm tra
- `n_features`: Số lượng features
- `n_classes`: Số lượng classes

## Sử dụng trong pipeline

Việc tạo dữ liệu được gọi qua MLflow thông qua entry point `generate_data`:

```bash
mlflow run . -e generate_data -P n_samples=2000 -P n_features=30 -P n_classes=3
```

Hoặc sử dụng giá trị mặc định:

```bash
mlflow run . -e generate_data
```

## Mở rộng

Ngoài việc sử dụng dữ liệu tổng hợp, module này có thể được mở rộng để:

1. **Tải dữ liệu thực tế**: Thêm các hàm để đọc từ CSV, Excel, hoặc các định dạng khác
2. **Tiền xử lý dữ liệu**: Thêm các hàm để chuẩn hóa, xử lý giá trị thiếu, mã hóa biến phân loại
3. **Augmentation dữ liệu**: Tạo thêm dữ liệu từ dữ liệu hiện có thông qua các kỹ thuật tăng cường 