import os
import random
import httpx
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

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
    phien_so_nguyen = Column(Integer, index=True, nullable=False) 
    
    open_time = Column(DateTime)
    ket_qua = Column(String) # "Tài" or "Xỉu"
    tong = Column(Integer)
    xuc_xac_1 = Column(Integer)
    xuc_xac_2 = Column(Integer)
    xuc_xac_3 = Column(Integer)
    created_at = Column(DateTime, default=datetime.now) 

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Tai Xiu Logic ---
def get_tai_xiu_result(xuc_xac_values: List[int]) -> Dict[str, any]:
    """Calculates Tai/Xiu result from 3 dice values."""
    if len(xuc_xac_values) != 3:
        raise ValueError("Phải có đúng 3 giá trị xúc xắc.")

    x1, x2, x3 = xuc_xac_values
    tong = x1 + x2 + x3
    ket_qua = "Tài" if 11 <= tong <= 17 else "Xỉu"

    # Rule for "Bão" (triple dice) - often classified as Xỉu
    if x1 == x2 == x3:
        ket_qua = "Xỉu" # Triplets (1-1-1 to 6-6-6) are often considered special "Xỉu"

    return {"Tong": tong, "Xuc_xac_1": x1, "Xuc_xac_2": x2, "Xuc_xac_3": x3, "Ket_qua": ket_qua}

# --- Pattern (Cầu) Analysis and Prediction Logic ---

def analyze_patterns_and_predict(historical_results: List[str]) -> Dict[str, str]:
    """
    Analyzes common "cầu" patterns in historical results and provides a prediction.
    This is a simplified example focusing on common patterns.
    """
    if len(historical_results) < 5: # Need at least a few results to find patterns
        return {"Ket_qua_du_doan": "Không đủ dữ liệu để phân tích cầu", "Ty_le_thang": "0%"}

    # Reverse the list so the most recent result is at index 0
    recent_history = historical_results[::-1] 
    
    # Simple pattern recognition
    # Cầu Bệt (Consecutive results)
    if len(recent_history) >= 3 and all(x == recent_history[0] for x in recent_history[:3]):
        # If the last 3 results are the same (e.g., Tài-Tài-Tài)
        # Predict the continuation of the "bệt"
        return {
            "Ket_qua_du_doan": recent_history[0], # Predicts Tài if last 3 were Tài
            "Ty_le_thang": "70% (theo cầu bệt)" # Higher confidence for strong patterns
        }

    # Cầu Đảo (Alternating results)
    if len(recent_history) >= 4 and \
       recent_history[0] != recent_history[1] and \
       recent_history[1] != recent_history[2] and \
       recent_history[2] != recent_history[3]:
        # If the last 4 results alternate (e.g., Tài-Xỉu-Tài-Xỉu)
        # Predict the continuation of the alternating pattern
        predicted = "Tài" if recent_history[0] == "Xỉu" else "Xỉu"
        return {
            "Ket_qua_du_doan": predicted, 
            "Ty_le_thang": "65% (theo cầu đảo)"
        }
    
    # Cầu 1-2-1 (Một Tài, hai Xỉu, một Tài)
    if len(recent_history) >= 4 and \
       recent_history[0] == recent_history[2] and \
       recent_history[1] == recent_history[3] and \
       recent_history[0] != recent_history[1]:
        # Example: Xỉu-Tài-Xỉu-Tài -> predict Xỉu
        # Example: Tài-Xỉu-Tài-Xỉu -> predict Tài
        predicted = recent_history[1] # Predict the one that appears twice
        return {
            "Ket_qua_du_doan": predicted,
            "Ty_le_thang": "60% (theo cầu 1-2-1)"
        }

    # Fallback to simple majority prediction if no specific pattern is found
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
                current_phien_record = db.query(PhienTaiXiu).filter(
                    PhienTaiXiu.phien_so_nguyen == phien_so_nguyen
                ).first()
                if not current_phien_record: 
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Lỗi hệ thống: Không thể lưu hoặc truy xuất phiên mới sau lỗi trùng lặp."
                    )
        else:
            current_phien_record = existing_phien

        # Fetch more historical sessions for better pattern analysis (e.g., 50 or 100)
        # I'll use 50 here. You can adjust this number.
        HISTORY_LIMIT = 50 
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
        
        # Extract only the "Ket_qua" for pattern analysis
        historical_outcomes = [p["Ket_qua"] for p in lich_su_formatted]

        # Predict based on pattern analysis
        prediction = analyze_patterns_and_predict(historical_outcomes)

        return {
            "Ket_qua": current_phien_record.ket_qua,
            "Phien": current_phien_record.expect_string,
            "Tong": current_phien_record.tong,
            "Xuc_xac_1": current_phien_record.xuc_xac_1,
            "Xuc_xac_2": current_phien_record.xuc_xac_2,
            "Xuc_xac_3": current_phien_record.xuc_xac_3,
            "id": "Nhutquang", 
            "Lich_su_gan_nhat": lich_su_formatted, # Renamed from "Lich_su_20_phien_gan_nhat" to be general
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

# To run locally: uvicorn main:app --reload
