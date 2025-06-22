import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
from datetime import datetime
import sys # Để thoát chương trình nếu cần

# Import các setup database và tính năng từ các file của bạn
# Đảm bảo các file này cùng cấp hoặc trong PYTHONPATH
from database import SessionLocal, PhienTaiXiu, Base, engine 
from features import create_training_data, FEATURE_COLUMNS # Import FEATURE_COLUMNS

def get_db_session():
    """Dependency để lấy session database cho script huấn luyện."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def train_and_save_model():
    print("--- Bắt đầu huấn luyện mô hình ---")

    # Đảm bảo bảng được tạo nếu chưa có
    print("Kiểm tra và tạo bảng database nếu chưa tồn tại...")
    Base.metadata.create_all(bind=engine)
    print("Hoàn thành kiểm tra bảng.")

    db_gen = get_db_session()
    db = next(db_gen) # Lấy đối tượng session từ generator

    try:
        # Lấy tất cả dữ liệu lịch sử từ database
        # Sắp xếp từ MỚI NHẤT đến CŨ NHẤT theo expect_string (ID phiên tăng dần)
        # Nếu expect_string không đại diện cho thời gian, hãy sắp xếp theo kai_jiang_time.desc()
        all_historical_records_db = db.query(PhienTaiXiu).order_by(PhienTaiXiu.expect_string.desc()).all()
        
        # Chỉ lấy cột 'ket_qua_phien' và chuyển thành list of strings
        all_historical_results_strings = [rec.ket_qua_phien for rec in all_historical_records_db if rec.ket_qua_phien]

        if not all_historical_results_strings:
            print("Không có dữ liệu lịch sử trong database để huấn luyện mô hình.")
            print("Vui lòng chạy ứng dụng FastAPI để thu thập dữ liệu trước khi huấn luyện.")
            sys.exit(1) # Thoát nếu không có dữ liệu

        print(f"Đã lấy {len(all_historical_results_strings)} bản ghi lịch sử từ database.")

        # Chuẩn bị dữ liệu huấn luyện (features X và labels y)
        X, y = create_training_data(all_historical_results_strings)

        if X.empty or y.empty:
            print("Không đủ dữ liệu sau khi tạo tính năng để huấn luyện mô hình.")
            print("Đảm bảo bạn có ít nhất 2 phiên lịch sử có kết quả để tạo 1 mẫu huấn luyện.")
            sys.exit(1) # Thoát nếu không đủ dữ liệu

        # Kiểm tra và đảm bảo các cột của X khớp với FEATURE_COLUMNS đã định nghĩa
        if list(X.columns) != FEATURE_COLUMNS:
            print(f"Lỗi nghiêm trọng: Thứ tự hoặc tên cột tính năng không khớp giữa features.py và dữ liệu tạo ra.")
            print(f"Cột thực tế: {list(X.columns)}")
            print(f"Cột mong đợi: {FEATURE_COLUMNS}")
            print("Vui lòng kiểm tra lại FEATURE_COLUMNS trong features.py và hàm extract_features.")
            sys.exit(1) # Thoát nếu cấu trúc cột không khớp

        print(f"Đã tạo {len(X)} mẫu huấn luyện/kiểm tra từ lịch sử.")
        print(f"Các tính năng được sử dụng: {X.columns.tolist()}")
        print(f"Các nhãn (kết quả) duy nhất: {y.unique().tolist()}")
        print(f"Phân bố nhãn: \n{y.value_counts(normalize=True)}")

        # Chia tập dữ liệu thành huấn luyện và kiểm tra
        # test_size=0.2: 20% dữ liệu dùng để kiểm tra mô hình
        # random_state=42: giúp kết quả chia tập dữ liệu được lặp lại (ổn định)
        # stratify=y: đảm bảo tỷ lệ Tài/Xỉu trong cả tập train và test là như nhau, quan trọng với dữ liệu không cân bằng
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"Kích thước tập huấn luyện: {len(X_train)} | Kích thước tập kiểm tra: {len(X_test)}")

        # Khởi tạo và huấn luyện mô hình Random Forest Classifier
        # n_estimators=200: Số cây quyết định trong rừng (tăng lên để có thể chính xác hơn)
        # class_weight='balanced': Giúp xử lý nếu số lượng Tài và Xỉu không cân bằng trong dữ liệu
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1) # n_jobs=-1 để dùng tất cả CPU core
        print("Bắt đầu huấn luyện mô hình Random Forest...")
        model.fit(X_train, y_train)
        print("Huấn luyện mô hình hoàn tất.")

        # Đánh giá mô hình trên tập kiểm tra
        print("\n--- Đánh giá mô hình trên tập kiểm tra ---")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_test, y_pred))
        print("\nMa trận nhầm lẫn (Confusion Matrix):")
        print(confusion_matrix(y_test, y_pred))

        # Lưu mô hình đã huấn luyện
        model_filename = 'model.pkl'
        joblib.dump(model, model_filename)
        print(f"\nMô hình đã được lưu vào file: {model_filename}")
        print("--- Hoàn thành huấn luyện mô hình ---")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình huấn luyện mô hình: {e}")
        sys.exit(1) # Thoát nếu có lỗi nghiêm trọng
    finally:
        db.close() # Đảm bảo đóng session

if __name__ == "__main__":
    # Đảm bảo biến môi trường DATABASE_URL được thiết lập khi chạy script này cục bộ
    # Nó phải trỏ đến database mà ứng dụng FastAPI của bạn đang lưu trữ dữ liệu.
    # Ví dụ: export DATABASE_URL="postgresql://user:password@host:port/dbname"
    
    if not os.getenv("DATABASE_URL"):
        print("Lỗi: Biến môi trường DATABASE_URL chưa được thiết lập.")
        print("Vui lòng thiết lập DATABASE_URL (trỏ đến database của bạn) trước khi chạy script huấn luyện.")
        sys.exit(1)
    
    train_and_save_model()

