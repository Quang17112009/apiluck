import os
import random
import httpx # Library for making HTTP requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Depends, HTTPException, status
# Đã thêm BigInteger vào import
from sqlalchemy import create_engine, Column, Integer, String, DateTime, BigInteger 
# Đã sửa lỗi cảnh báo: import declarative_base từ sqlalchemy.orm thay vì sqlalchemy.ext.declarative
from sqlalchemy.orm import sessionmaker, Session, declarative_base 
from sqlalchemy.exc import IntegrityError # To handle duplicate key errors

app = FastAPI()

# --- Database Configuration (PostgreSQL) ---
# Lấy chuỗi kết nối từ biến môi trường (cho Render)
# Fallback (dự phòng) về một chuỗi kết nối PostgreSQL cục bộ hoặc SQLite cho phát triển cục bộ
# Nếu bạn chạy cục bộ với PostgreSQL, hãy thay đổi "postgresql://user:password@localhost/taixiu_db" cho phù hợp
# Nếu bạn muốn chạy cục bộ với SQLite và chấp nhận mất dữ liệu khi ứng dụng khởi động lại, hãy dùng: "sqlite:///./taixiu.db"
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/taixiu_db") 

# Tạo engine và session maker
engine = create_engine(SQLALCHEMY_DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Model Definition ---
class PhienTaiXiu(Base):
    __tablename__ = "phien_tai_xiu"
    
    id = Column(Integer, primary_key=True, index=True)
    # expect_string từ API bên ngoài, dùng làm định danh duy nhất cho một phiên
    expect_string = Column(String, unique=True, index=True, nullable=False) 
    # ĐÃ THAY ĐỔI KIỂU DỮ LIỆU TỪ Integer SANG BigInteger
    phien_so_nguyen = Column(BigInteger, index=True, nullable=False) 
    
    open_time = Column(DateTime)
    ket_qua = Column(String) # "Tài" or "Xỉu"
    tong = Column(Integer)
    xuc_xac_1 = Column(Integer)
    xuc_xac_2 = Column(Integer)
    xuc_xac_3 = Column(Integer)
    created_at = Column(DateTime, default=datetime.now) # Thời gian record được tạo trong DB của chúng ta

# --- KHỞI TẠO BẢNG DATABASE (QUAN TRỌNG) ---
# Uncomment dòng này VÀ CHẠY ỨNG DỤNG MỘT LẦN ĐỂ TẠO BẢNG trong cơ sở dữ liệu PostgreSQL của bạn.
# SAU KHI BẢNG ĐƯỢC TẠO THÀNH CÔNG, HÃY COMMENT LẠI dòng này và triển khai lại.
# Nếu bạn đang sử dụng SQLite, điều này cũng sẽ tạo file .db
Base.metadata.create_all(bind=engine)

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

# --- Logic phân tích "Cầu" và Dự đoán ---
def analyze_patterns_and_predict(historical_results: List[str]) -> Dict[str, str]:
    """
    Phân tích các mẫu "cầu" phổ biến trong lịch sử và đưa ra dự đoán.
    Đây là một ví dụ đơn giản tập trung vào các mẫu cầu cơ bản.
    """
    # Cần ít nhất 5 kết quả để có thể tìm kiếm các mẫu cầu cơ bản
    if len(historical_results) < 5: 
        return {"Ket_qua_du_doan": "Không đủ dữ liệu để phân tích cầu", "Ty_le_thang": "0%"}

    # Đảo ngược danh sách để kết quả gần nhất nằm ở đầu (index 0)
    recent_history = historical_results[::-1] 
    
    # --- Nhận diện các mẫu cầu ---

    # Cầu Bệt (Consecutive results) - ví dụ: Tài-Tài-Tài
    # Kiểm tra 3, 4, 5... kết quả gần nhất
    for i in range(3, len(recent_history) + 1):
        current_sequence = recent_history[:i]
        if all(x == current_sequence[0] for x in current_sequence):
            # Nếu tìm thấy cầu bệt dài i
            return {
                "Ket_qua_du_doan": current_sequence[0], # Dự đoán tiếp tục bệt
                "Ty_le_thang": f"{min(85, 50 + i * 5)}% (theo cầu bệt dài {i})" # Tỷ lệ tăng theo độ dài bệt
            }

    # Cầu Đảo (Alternating results) - ví dụ: Tài-Xỉu-Tài-Xỉu
    if len(recent_history) >= 4:
        is_alternating = True
        for i in range(len(recent_history) - 1):
            if recent_history[i] == recent_history[i+1]:
                is_alternating = False
                break
        if is_alternating:
            predicted = "Tài" if recent_history[0] == "Xỉu" else "Xỉu"
            return {
                "Ket_qua_du_doan": predicted, 
                "Ty_le_thang": "65% (theo cầu đảo)"
            }
    
    # Cầu 1-2-1 / 1-1-2 / 2-1-1 (Một Tài, hai Xỉu, một Tài / Một Xỉu, hai Tài, một Xỉu)
    # Ví dụ: Tài-Xỉu-Xỉu-Tài -> dự đoán Xỉu (vì có 2 Xỉu ở giữa)
    if len(recent_history) >= 4:
        # Mẫu A-B-B-A
        if recent_history[0] == recent_history[3] and \
           recent_history[1] == recent_history[2] and \
           recent_history[0] != recent_history[1]:
            # VD: Tài-Xỉu-Xỉu-Tài -> Dự đoán Xỉu (Tiếp tục cặp đôi ở giữa)
            return {
                "Ket_qua_du_doan": recent_history[1],
                "Ty_le_thang": "60% (theo cầu 1-2-1/2-1-1)"
            }
        # Mẫu A-B-A-A
        if recent_history[0] == recent_history[2] == recent_history[3] and \
           recent_history[0] != recent_history[1]:
            # VD: Tài-Xỉu-Tài-Tài -> Dự đoán Xỉu (Để tạo thành 2 Tài - 2 Xỉu)
            return {
                "Ket_qua_du_doan": recent_history[1],
                "Ty_le_thang": "58% (theo cầu 1-1-2)"
            }


    # --- Dự phòng: Dự đoán theo đa số nếu không tìm thấy cầu cụ thể ---
    tai_count = historical_results.count("Tài")
    xiu_count = historical_results.count("Xỉu")
    total_count = len(historical_results)

    if total_count == 0:
        return {"Ket_qua_du_doan": "Không có dữ liệu", "Ty_le_thang": "0%"}

    if tai_count > xiu_count:
        predicted_outcome = "Tài"
        win_percentage = f"{(tai_count / total_count) * 100:.0f}%"
    elif xiu_count > tai_count:
        predicted_outcome = "Xỉu"
        win_percentage = f"{(xiu_count / total_count) * 100:.0f}%"
    else:
        # Nếu số lượng Tài và Xỉu bằng nhau, dự đoán ngẫu nhiên
        predicted_outcome = random.choice(["Tài", "Xỉu"])
        win_percentage = "50%"

    return {"Ket_qua_du_doan": predicted_outcome, "Ty_le_thang": f"{win_percentage} (thống kê chung)"}


# --- Main API Endpoint ---
@app.get("/api/taixiu")
async def get_taixiu_data_with_history_and_prediction(db: Session = Depends(get_db)):
    EXTERNAL_API_URL = "https://1.bot/GetNewLottery/LT_Taixiu"
    
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
        phien_so_nguyen = int(expect_str) 
        
        open_code_str = data["OpenCode"]
        xuc_xac_values = [int(x.strip()) for x in open_code_str.split(',')]
        
        open_time_str = data["OpenTime"]
        open_time_dt = datetime.strptime(open_time_str, "%Y-%m-%d %H:%M:%S")

        current_result_data = get_tai_xiu_result(xuc_xac_values)
        
        current_phien_record: Optional[PhienTaiXiu] = None

        existing_phien = db.query(PhienTaiXiu).filter(
            PhienTaiXiu.phien_so_nguyen == phien_so_nguyen
        ).first()
        
        if not existing_phien:
            new_phien = PhienTaiXiu(
                phien_so_nguyen=phien_so_nguyen,
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
                # Nếu có lỗi trùng lặp, có thể do một request khác đã thêm vào trước
                # Cố gắng truy vấn lại để lấy phiên đã tồn tại
                current_phien_record = db.query(PhienTaiXiu).filter(
                    PhienTaiXiu.phien_so_nguyen == phien_so_nguyen
                ).first()
                if not current_phien_record: 
                    # Nếu vẫn không tìm thấy, có lỗi nghiêm trọng hơn
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Lỗi hệ thống: Không thể lưu hoặc truy xuất phiên mới sau lỗi trùng lặp."
                    )
        else:
            current_phien_record = existing_phien

        # Lấy số lượng phiên lịch sử lớn hơn để phân tích cầu (ví dụ: 100 phiên)
        HISTORY_LIMIT = 100 
        lich_su = db.query(PhienTaiXiu).order_by(PhienTaiXiu.phien_so_nguyen.desc()).limit(HISTORY_LIMIT).all()
        
        lich_su_formatted = [
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
        
        # Chỉ lấy kết quả "Tài" hoặc "Xỉu" để truyền vào hàm phân tích cầu
        historical_outcomes = [p["Ket_qua"] for p in lich_su_formatted]

        # Dự đoán dựa trên phân tích cầu
        prediction = analyze_patterns_and_predict(historical_outcomes)

        # Trả về phản hồi API cuối cùng
        return {
            "Ket_qua": current_phien_record.ket_qua,
            "Phien": current_phien_record.expect_string,
            "Tong": current_phien_record.tong,
            "Xuc_xac_1": current_phien_record.xuc_xac_1,
            "Xuc_xac_2": current_phien_record.xuc_xac_2,
            "Xuc_xac_3": current_phien_record.xuc_xac_3,
            "id": "Nhutquang", 
            "Lich_su_gan_nhat": lich_su_formatted, 
            "Du_doan_phien_tiep_theo": prediction
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

# Để chạy cục bộ:
# uvicorn main:app --reload
