# attitude-analysis-vietnamese
Bài tập lớn môn Nhập môn Trí tuệ Nhân tạo: Xây dựng mô hình phân loại thái độ văn bản tiếng Việt sử dụng PhoBERT, BiLSTM, Attention.
Số điểm nhóm đạt được: 9/10

## Hướng dẫn chạy dự án

### 1. Tạo virtual environment (khuyến nghị)

Mở terminal tại thư mục dự án và chạy:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Cài đặt các thư viện cần thiết

```powershell
pip install -r requirements.txt
```

### 3. Chạy giao diện phân tích thái độ

```powershell
python main.py
```

### 4. Cấu trúc thư mục
- `trained_model/`: Chứa model, label encoder, model info đã huấn luyện.
- `main.py`: File giao diện phân tích thái độ.
- `requirements.txt`: Danh sách thư viện cần thiết.
- `dataset/`: Dữ liệu gốc.
- `notebook/`: Notebook huấn luyện và kiểm thử.
- `test model`: File notebook dùng test và link kaggle nếu muốn test online.

