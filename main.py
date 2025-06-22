import os
import random
import httpx
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import IntegrityError

# --- Machine Learning Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = FastAPI()

# --- Database Configuration (PostgreSQL) ---
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/taixiu_db")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Model Definition ---
class PhienTaiXiu(Base):
    __tablename__ = "phien_tai_xiu"

    id = Column(Integer, primary_key=True, index=True)
    expect_string = Column(String, unique=True, index=True, nullable=False)
    open_time = Column(DateTime)
    ket_qua = Column(String) # "Tài" or "Xỉu"
    tong = Column(Integer)
    xuc_xac_1 = Column(Integer)
    xuc_xac_2 = Column(Integer)
    xuc_xac_3 = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)

# --- KHỞI TẠO BẢNG DATABASE (QUAN TRỌNG) ---
# Uncomment dòng này VÀ CHẠY ỨNG DỤNG MỘT LẦN ĐỂ TẠO BẢNG trong cơ sở dữ liệu PostgreSQL của bạn.
# SAU KHI BẢNG ĐƯỢC TẠO THÀNH CÔNG, HÃY COMMENT LẠI dòng này và triển khai lại.
# Base.metadata.create_all(bind=engine)

# Dependency để lấy Session DB cho mỗi request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Logic tính Tài Xỉu ---
def get_tai_xiu_result(xuc_xac_values: List[int]) -> Dict[str, any]:
    """Tính toán kết quả Tài/Xỉu từ 3 giá trị xúc xắc."""
    if len(xuc_xac_values) != 3:
        raise ValueError("Phải có đúng 3 giá trị xúc xắc.")

    x1, x2, x3 = xuc_xac_values
    tong = x1 + x2 + x3
    ket_qua = "Tài" if 11 <= tong <= 17 else "Xỉu" # Tài: 11-17, Xỉu: 4-10

    # Quy tắc cho "Bão" (bộ 3 đồng nhất) - thường được coi là Xỉu
    if x1 == x2 == x3:
        ket_qua = "Xỉu" # Bộ 3 đồng nhất (ví dụ: 1-1-1, 6-6-6) được coi là Xỉu

    return {"Tong": tong, "Xuc_xac_1": x1, "Xuc_xac_2": x2, "Xuc_xac_3": x3, "Ket_qua": ket_qua}

# --- Machine Learning Model for Prediction ---
def predict_with_ml_model(historical_results: List[str]) -> Dict[str, str]:
    """
    Sử dụng mô hình học máy để dự đoán kết quả Tài/Xỉu và độ tin cậy.
    """
    if len(historical_results) < 20: # Cần nhiều dữ liệu hơn để huấn luyện mô hình
        return {"Ket_qua_du_doan": "Không đủ dữ liệu để huấn luyện ML", "Do_tin_cay": "N/A"}

    # Encode "Tài" và "Xỉu" thành số
    le = LabelEncoder()
    encoded_results = le.fit_transform(historical_results) # Tài -> 1, Xỉu -> 0 (hoặc ngược lại)

    # Chuẩn bị dữ liệu cho mô hình
    # Chúng ta sẽ sử dụng một "cửa sổ" các kết quả trước đó để dự đoán kết quả tiếp theo.
    # Ví dụ: dùng 5 kết quả trước để dự đoán kết quả thứ 6.
    window_size = 5 # Số lượng kết quả lịch sử để xem xét cho mỗi dự đoán
    X = [] # Features
    y = [] # Labels

    for i in range(len(encoded_results) - window_size):
        X.append(encoded_results[i : i + window_size])
        y.append(encoded_results[i + window_size])

    if not X: # Trường hợp không đủ dữ liệu sau khi tạo cửa sổ
        return {"Ket_qua_du_doan": "Không đủ dữ liệu sau khi tạo cửa sổ ML", "Do_tin_cay": "N/A"}

    X = np.array(X)
    y = np.array(y)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (tùy chọn, ở đây ta huấn luyện trên toàn bộ)
    # For a simple online prediction, we might train on all available data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X, y)

        # Dự đoán cho phiên tiếp theo
        # Lấy 5 kết quả gần nhất để làm đầu vào cho dự đoán
        last_n_results = encoded_results[-window_size:].reshape(1, -1)
        
        predicted_encoded = model.predict(last_n_results)[0]
        predicted_outcome = le.inverse_transform([predicted_encoded])[0]

        # Lấy xác suất dự đoán
        probabilities = model.predict_proba(last_n_results)[0]
        confidence = probabilities[model.classes_ == predicted_encoded][0] * 100 # Chuyển đổi thành %

        return {
            "Ket_qua_du_doan": predicted_outcome,
            "Do_tin_cay": f"{confidence:.2f}%"
        }
    except Exception as e:
        print(f"Error during ML prediction: {e}")
        return {"Ket_qua_du_doan": "Lỗi khi chạy mô hình ML", "Do_tin_cay": "N/A"}


# --- Main API Endpoint ---
@app.get("/api/taixiu")
async def get_taixiu_data_with_history_and_prediction(db: Session = Depends(get_db)):
    EXTERNAL_API_URL = "https://1.bot/GetNewLottery/LT_Taixiu" # This URL is likely a placeholder/example

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(EXTERNAL_API_URL, timeout=10.0)
            response.raise_for_status()
            external_data = response.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi kết nối đến API bên ngoài: {exc}. Vui lòng thử lại sau."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý phản hồi từ API bên ngoài: {e}"
        )

    if external_data.get("state") != 1 or not external_data.get("data"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dữ liệu từ API bên ngoài không hợp lệ hoặc không có kết quả."
        )

    data = external_data["data"]

    try:
        expect_str = str(data["Expect"])

        open_code_str = data["OpenCode"]
        xuc_xac_values = [int(x.strip()) for x in open_code_str.split(',')]

        open_time_str = data["OpenTime"]
        open_time_dt = datetime.strptime(open_time_str, "%Y-%m-%d %H:%M:%S")

        current_result_data = get_tai_xiu_result(xuc_xac_values)

        current_phien_record: Optional[PhienTaiXiu] = None

        existing_phien = db.query(PhienTaiXiu).filter(
            PhienTaiXiu.expect_string == expect_str
        ).first()

        if not existing_phien:
            new_phien = PhienTaiXiu(
                expect_string=expect_str,
                open_time=open_time_dt,
                ket_qua=current_result_data["Ket_qua"],
                tong=current_result_data["Tong"],
                xuc_xac_1=current_result_data["Xuc_xac_1"],
                xuc_xac_2=current_result_data["Xuc_xac_2"],
                xuc_xac_3=current_result_data["Xuc_xac_3"]
            )
            db.add(new_phien)
            try:
                db.commit()
                db.refresh(new_phien)
                current_phien_record = new_phien
            except IntegrityError:
                db.rollback()
                current_phien_record = db.query(PhienTaiXiu).filter(
                    PhienTaiXiu.expect_string == expect_str
                ).first()
                if not current_phien_record:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Lỗi hệ thống: Không thể lưu hoặc truy xuất phiên mới sau lỗi trùng lặp."
                    )
        else:
            current_phien_record = existing_phien

        # Lấy số lượng phiên lịch sử lớn hơn để phân tích cầu và ML (ví dụ: 200 phiên trở lên)
        HISTORY_LIMIT_FOR_ANALYSIS = 200
        # Chỉ hiển thị 20 phiên gần nhất trong phản hồi API
        DISPLAY_HISTORY_LIMIT = 20

        # Truy vấn các phiên để đảm bảo dữ liệu phân tích đầy đủ
        lich_su = db.query(PhienTaiXiu).order_by(PhienTaiXiu.expect_string.desc()).limit(HISTORY_LIMIT_FOR_ANALYSIS).all()

        # Chỉ định dạng và lấy 20 phiên đầu tiên cho phần hiển thị
        lich_su_formatted_full = [
            {
                "Phien": p.expect_string,
                "Ket_qua": p.ket_qua,
                "Tong": p.tong,
                "Xuc_xac_1": p.xuc_xac_1,
                "Xuc_xac_2": p.xuc_xac_2,
                "Xuc_xac_3": p.xuc_xac_3,
                "OpenTime": p.open_time.strftime("%Y-%m-%d %H:%M:%S")
            } for p in lich_su
        ]
        # Cắt lấy 20 phiên gần nhất để hiển thị
        lich_su_formatted_display = lich_su_formatted_full[:DISPLAY_HISTORY_LIMIT]

        # Chỉ lấy kết quả "Tài" hoặc "Xỉu" từ TẤT CẢ các phiên để truyền vào hàm ML
        historical_outcomes_for_analysis = [p["Ket_qua"] for p in lich_su_formatted_full]

        # Dự đoán dựa trên mô hình học máy
        ml_prediction = predict_with_ml_model(historical_outcomes_for_analysis)

        # Trả về phản hồi API cuối cùng
        return {
            "Ket_qua_phien_hien_tai": current_phien_record.ket_qua,
            "Ma_phien_hien_tai": current_phien_record.expect_string,
            "Tong_diem_hien_tai": current_phien_record.tong,
            "Xuc_xac_hien_tai": [
                current_phien_record.xuc_xac_1,
                current_phien_record.xuc_xac_2,
                current_phien_record.xuc_xac_3
            ],
            "admin_name": "Nhutquang",
            "Lich_su_gan_nhat": lich_su_formatted_display,
            "Du_doan_phien_tiep_theo_ML": ml_prediction # Đã đổi tên để rõ ràng hơn
        }

    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dữ liệu từ API bên ngoài không đúng định dạng hoặc thiếu trường bắt buộc: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi không xác định trong quá trình xử lý yêu cầu: {e}"
        )

