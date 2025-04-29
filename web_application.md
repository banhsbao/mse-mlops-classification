# Ứng dụng Web

## Tổng quan

Ứng dụng web được xây dựng bằng Flask để cho phép người dùng tương tác với mô hình đã huấn luyện. Ứng dụng cung cấp giao diện thân thiện cho việc nhập dữ liệu và hiển thị kết quả dự đoán.

## Cấu trúc ứng dụng

```
app/
├── app.py                  # Flask application
├── templates/              # HTML templates
│   ├── index.html          # Home page
│   └── result.html         # Results page
└── static/                 # Static assets
    ├── css/                # Stylesheets
    └── js/                 # JavaScript files
```

## Tính năng chính

1. **Trang chủ (index.html)**
   - Form nhập dữ liệu với các tùy chọn cho số lượng features
   - Hỗ trợ tải lên tập dữ liệu (CSV)
   - Hướng dẫn về định dạng dữ liệu đầu vào
   - Responsive design thích ứng với nhiều thiết bị

2. **Trang kết quả (result.html)**
   - Hiển thị kết quả dự đoán (classes và xác suất)
   - Biểu đồ trực quan hóa kết quả
   - Tùy chọn để download kết quả
   - Liên kết quay lại để thực hiện dự đoán mới

## Mã nguồn

### Flask Application (app.py)

```python
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import MLPClassifier

app = Flask(__name__)

# Load the trained model
def load_model():
    model_path = '../models/mlp_classifier.pt'
    with open('../data/classification_data.pkl', 'rb') as f:
        data_info = pickle.load(f)
    
    n_features = data_info['n_features']
    n_classes = data_info['n_classes']
    
    model = MLPClassifier(
        input_dim=n_features,
        output_dim=n_classes
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, n_features, n_classes

model, n_features, n_classes = load_model()

@app.route('/')
def index():
    return render_template('index.html', n_features=n_features)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # Handle file upload
        file = request.files['file']
        data = np.loadtxt(file, delimiter=',')
    else:
        # Handle form input
        features = []
        for i in range(n_features):
            feature_val = float(request.form.get(f'feature_{i}', 0))
            features.append(feature_val)
        data = np.array([features])
    
    # Convert to tensor and make prediction
    input_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        pred_class = model.predict(input_tensor).numpy()
        pred_proba = model.predict_proba(input_tensor).numpy()
    
    # Format results
    results = []
    for i in range(len(pred_class)):
        proba_dict = {str(j): float(pred_proba[i][j]) for j in range(n_classes)}
        results.append({
            'instance': i+1,
            'predicted_class': int(pred_class[i]),
            'probabilities': proba_dict
        })
    
    return render_template('result.html', results=results, n_classes=n_classes)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
```

### HTML Templates

Các templates sử dụng HTML5, CSS, và JavaScript để tạo giao diện người dùng trực quan và dễ sử dụng.

## Triển khai

Ứng dụng có thể được chạy như một bước trong pipeline MLflow:

```bash
mlflow run . -e app
```

Hoặc khởi chạy trực tiếp:

```bash
python app/app.py
```

Ứng dụng mặc định chạy trên cổng 8000 và có thể được truy cập tại `http://localhost:8000`.

## Tương thích thiết bị

Ứng dụng được thiết kế responsive để tương thích với nhiều thiết bị:

1. **Desktop**
   - Giao diện đầy đủ với tất cả các tính năng
   - Hiển thị biểu đồ chi tiết

2. **Tablet**
   - Giao diện thích ứng với màn hình trung bình
   - Bố cục tối ưu cho các thiết bị cầm tay

3. **Smartphone**
   - Giao diện thu gọn cho màn hình nhỏ
   - Các form và controls được điều chỉnh kích thước phù hợp
   - Ưu tiên hiển thị thông tin quan trọng nhất

## Bảo mật

Ứng dụng bao gồm các biện pháp bảo mật cơ bản:

1. Validation đầu vào
2. Xử lý lỗi và các trường hợp ngoại lệ
3. Giới hạn kích thước tập tin tải lên 