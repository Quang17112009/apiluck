import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
from datetime import datetime
import sys

# Đã thay đổi từ .database thành database để khắc phục ImportError
from database import SessionLocal, PhienTaiXiu, Base, engine
# Đã thay đổi từ .features thành features để khắc phục ImportError
from features import create_training_data, FEATURE_COLUMNS

def get_db_session():
    """Dependency để lấy session database cho script huấn luyện."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def train_and_save_model():
    print("--- Bắt đầu huấn luyện mô hình ---")

    print("Kiểm tra và tạo bảng database nếu chưa tồn tại...")
    Base.metadata.create_all(bind=engine)
    print("Hoàn thành kiểm tra bảng.")

    db_gen = get_db_session()
    db = next(db_gen)

    try:
        # Lấy tất cả dữ liệu lịch sử từ database
        # Sử dụng kai_jiang_time để sắp xếp lịch sử từ MỚI NHẤT -> CŨ NHẤT
        all_historical_records_db = db.query(PhienTaiXiu).order_by(PhienTaiXiu.kai_jiang_time.desc()).all()
        all_historical_results_strings = [rec.ket_qua_phien for rec in all_historical_records_db if rec.ket_qua_phien]

        if not all_historical_results_strings:
            print("Không có dữ liệu lịch sử trong database để huấn luyện mô hình.")
            print("Vui lòng chạy ứng dụng FastAPI để thu thập dữ liệu trước khi huấn luyện.")
            sys.exit(1)

        print(f"Đã lấy {len(all_historical_results_strings)} bản ghi lịch sử từ database.")

        X, y = create_training_data(all_historical_results_strings)

        if X.empty or y.empty:
            print("Không đủ dữ liệu sau khi tạo tính năng để huấn luyện mô hình.")
            print("Đảm bảo bạn có ít nhất 2 phiên lịch sử có kết quả để tạo 1 mẫu huấn luyện.")
            sys.exit(1)

        if list(X.columns) != FEATURE_COLUMNS:
            print(f"Lỗi nghiêm trọng: Thứ tự hoặc tên cột tính năng không khớp giữa features.py và dữ liệu tạo ra.")
            print(f"Cột thực tế: {list(X.columns)}")
            print(f"Cột mong đợi: {FEATURE_COLUMNS}")
            print("Vui lòng kiểm tra lại FEATURE_COLUMNS trong features.py và hàm extract_features.")
            sys.exit(1)

        print(f"Đã tạo {len(X)} mẫu huấn luyện/kiểm tra từ lịch sử.")
        print(f"Các tính năng được sử dụng: {X.columns.tolist()}")
        print(f"Các nhãn (kết quả) duy nhất: {y.unique().tolist()}")
        print(f"Phân bố nhãn: \n{y.value_counts(normalize=True)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"Kích thước tập huấn luyện: {len(X_train)} | Kích thước tập kiểm tra: {len(X_test)}")

        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
        print("Bắt đầu huấn luyện mô hình Random Forest...")
        model.fit(X_train, y_train)
        print("Huấn luyện mô hình hoàn tất.")

        print("\n--- Đánh giá mô hình trên tập kiểm tra ---")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_test, y_pred))
        print("\nMa trận nhầm lẫn (Confusion Matrix):")
        print(confusion_matrix(y_test, y_pred))

        model_filename = 'model.pkl'
        joblib.dump(model, model_filename)
        print(f"\nMô hình đã được lưu vào file: {model_filename}")
        print("--- Hoàn thành huấn luyện mô hình ---")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình huấn luyện mô hình: {e}")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    if not os.getenv("DATABASE_URL"):
        print("Lỗi: Biến môi trường DATABASE_URL chưa được thiết lập.")
        print("Vui lòng thiết lập DATABASE_URL (trỏ đến database của bạn) trước khi chạy script huấn luyện.")
        sys.exit(1)
    
    train_and_save_model()
