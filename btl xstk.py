import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import sys

# Cấu hình UTF-8 cho stdout
sys.stdout.reconfigure(encoding='utf-8')

#Bước 1: Làm sạch dữ liệu
print("--- 3.2 Làm sạch dữ liệu ---")

# Đọc dữ liệu từ file Excel
file_path = r'C:\Users\thiho\Downloads\BTL_SXTK\MucLuong_2021_2024.xlsx'
df = pd.read_excel(file_path)

# Hiển thị danh sách cột và 20 dòng đầu tiên để kiểm tra
print("Danh sách cột trong file:")
print(df.columns.tolist())
print("\nDữ liệu ban đầu:")
print(df.head(20))

# Phát hiện các vấn đề
print("\nPhát hiện vấn đề:")
print("Giá trị thiếu trong từng cột:")
print(df.isnull().sum())

# 1. Điền giá trị thiếu
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print("\nSau khi điền giá trị thiếu:")
print(df.isnull().sum())


# 3. Chuẩn hóa dữ liệu
# Chuẩn hóa cột "Vị trí"
df['Vị trí'] = df['Vị trí'].str.capitalize()
# Sửa lỗi chính tả trong cột "Ngành nghề"
df['Ngành nghề'] = df['Ngành nghề'].replace('Mô giới chứng khoán', 'Môi giới chứng khoán')
# Đảm bảo các cột lương là số nguyên
cols = ['Lương TB Q1', 'Lương TB Q2', 'Lương TB Q3', 'Lương TB Q4']
for col in cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Tạo cột Lương TB (trung bình của 4 quý)
df['Lương TB'] = df[cols].mean(axis=1)
print("\nDữ liệu sau khi làm sạch:")
print(df.head(20))

# 3.3 Khám phá dữ liệu
print("\n---Khám phá dữ liệu ---")

# 1. Thống kê cơ bản
salary_by_year = df.groupby('Năm')['Lương TB'].mean() / 1e6
print("\nLương trung bình theo năm (triệu VND):")
print(salary_by_year)

salary_by_nganh = df.groupby('Ngành nghề')['Lương TB'].mean() / 1e6
print("\nLương trung bình theo ngành nghề (triệu VND):")
print(salary_by_nganh)

# Số lượng bản ghi theo vị trí
print("\nSố lượng bản ghi theo vị trí:")
print(df['Vị trí'].value_counts())

# 3. Biểu đồ theo thời gian (xu hướng lương của tất cả các ngành nghề)
plt.figure(figsize=(12, 6))
for nganh in df['Ngành nghề'].unique():
    nganh_data = df[df['Ngành nghề'] == nganh].groupby('Năm')['Lương TB'].mean() / 1e6
    plt.plot(nganh_data.index, nganh_data, marker='o', label=nganh)
plt.title('Xu hướng lương trung bình theo năm của tất cả các ngành nghề')
plt.xlabel('Năm')
plt.ylabel('Lương trung bình (triệu VND)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Phân tích theo ngành nghề
plt.figure(figsize=(12, 6))
salary_by_nganh_plot = df.groupby('Ngành nghề')['Lương TB'].mean() / 1e6
salary_by_nganh_plot.plot(kind='bar', color='lightgreen')
plt.title('Lương trung bình theo ngành nghề (triệu VND)')
plt.xlabel('Ngành nghề')
plt.ylabel('Lương trung bình')
plt.xticks(rotation=45)
plt.show()

# 3.4 Xây dựng mô hình
print("\n--- 3.4 Xây dựng mô hình ---")

# Chuẩn bị dữ liệu cho mô hình đa biến
features = ['Ngành nghề', 'Vị trí', 'Năm']
target = 'Lương TB'

# Mã hóa one-hot cho các cột phân loại
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[features])

# Chọn biến đầu vào và đầu ra
X_multi = encoded_features
y_multi = df[target]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Xây dựng mô hình Linear Regression đa biến
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

# Đánh giá mô hình đa biến
y_pred_multi = model_multi.predict(X_test_multi)
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)
print("\nKết quả đánh giá mô hình Linear Regression đa biến:")
print(f'MSE: {mse_multi}')
print(f'R-squared: {r2_multi}')

# Thêm hồi quy tuyến tính đơn biến (dựa trên cột Năm)
X_simple = df[['Năm']].values
y_simple = df['Lương TB'].values

# Chia dữ liệu
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Xây dựng mô hình Linear Regression đơn biến
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)

# Đánh giá mô hình đơn biến
y_pred_simple = model_simple.predict(X_test_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)
print("\nKết quả đánh giá mô hình Linear Regression đơn biến (dựa trên Năm):")
print(f'MSE: {mse_simple}')
print(f'R-squared: {r2_simple}')

# 3.5 Trực quan hóa dữ liệu
print("\n--- 3.5 Trực quan hóa dữ liệu ---")

# Dự đoán lương cho tất cả các ngành nghề từ 2025-2030
future_years = np.array(range(2025, 2031)).reshape(-1, 1)
predictions_by_nganh = {}

for nganh in df['Ngành nghề'].unique():
        df_nganh = df[df['Ngành nghề'] == nganh].copy()
X_nganh = df_nganh[['Năm']].values
y_nganh = df_nganh['Lương TB'].values
    
    # Huấn luyện mô hình Linear Regression cho từng ngành
model_nganh = LinearRegression()
model_nganh.fit(X_nganh, y_nganh)
    
    # Dự đoán
future_predictions = model_nganh.predict(future_years)
predictions_by_nganh[nganh] = future_predictions