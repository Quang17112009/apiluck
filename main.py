import os
import asyncio
import random
import httpx
import joblib
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Đã thay đổi từ .database thành database để khắc phục ImportError
from database import Base, get_db, PhienTaiXiu
# Đã thay đổi từ .features thành features để khắc phục ImportError
from features import extract_features, FEATURE_COLUMNS

# --- Cấu hình Database ---
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please set it for Render deployment.")

engine = create_engine(DATABASE_URL)
# Đảm bảo bảng được tạo nếu chưa có.
Base.metadata.create_all(bind=engine) 

app = FastAPI()

# --- Tải Mô hình Học máy ---
ml_model = None
MODEL_PATH = 'model.pkl'

try:
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
        print("Mô hình học máy đã được tải thành công.")
    else:
        print(f"Cảnh báo: Không tìm thấy file mô hình '{MODEL_PATH}'. Dự đoán sẽ sử dụng logic thống kê cũ.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}. Dự đoán sẽ sử dụng logic thống kê cũ.")
    ml_model = None

# --- Hằng số ---
EXTERNAL_API_URL = "https://www.1.bot/api/data?type=cq"
FETCH_INTERVAL_SECONDS = 5
HISTORY_LIMIT_FOR_ANALYSIS = 100

# --- Background Task để fetch dữ liệu từ API ngoài và lưu vào DB ---
async def fetch_data_from_external_api_in_background():
    print("Bắt đầu tác vụ nền fetch dữ liệu...")
    while True:
        db: Session = next(get_db())
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(EXTERNAL_API_URL, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                for item in data.get("list", []):
                    expect_string = item.get("expect")
                    open_code = item.get("openCode")
                    kai_jiang_time_str = item.get("kaiJiangTime")

                    if not expect_string or not open_code or not kai_jiang_time_str:
                        continue

                    try:
                        # API có thể trả về timestamp dưới dạng chuỗi, cần đảm bảo đúng kiểu
                        kai_jiang_time = datetime.fromtimestamp(int(kai_jiang_time_str) / 1000)
                    except (ValueError, TypeError):
                        print(f"Lỗi chuyển đổi kaiJiangTime: {kai_jiang_time_str}. Bỏ qua.")
                        continue

                    existing_phien = db.query(PhienTaiXiu).filter(PhienTaiXiu.expect_string == expect_string).first()
                    if existing_phien:
                        continue

                    try:
                        dice_results = [int(d) for d in open_code.split(',') if d.strip().isdigit()]
                        if len(dice_results) != 3:
                            raise ValueError("openCode không có đủ 3 viên xúc xắc hợp lệ.")
                        
                        tong_diem = sum(dice_results)
                        ket_qua_phien = "Tài" if tong_diem >= 11 else "Xỉu"
                    except (ValueError, IndexError):
                        print(f"Lỗi phân tích openCode: '{open_code}'. Bỏ qua.")
                        continue

                    new_phien = PhienTaiXiu(
                        expect_string=expect_string,
                        open_code=open_code,
                        kai_jiang_time=kai_jiang_time,
                        ket_qua_phien=ket_qua_phien,
                        tong_diem=tong_diem,
                    )
                    db.add(new_phien)
                    db.commit()
                    db.refresh(new_phien)
                    print(f"Đã lưu phiên {new_phien.expect_string} vào DB. Kết quả: {new_phien.ket_qua_phien}, Tổng: {new_phien.tong_diem}")

        except httpx.RequestError as exc:
            print(f"Lỗi kết nối API ngoài: {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"Lỗi HTTP từ API ngoài: Status {exc.response.status_code} - {exc.response.text}")
        except Exception as e:
            print(f"Lỗi không xác định trong tác vụ nền fetch_data: {e}")
        finally:
            db.close()

        await asyncio.sleep(FETCH_INTERVAL_SECONDS)

# --- Khởi chạy Background Task khi ứng dụng khởi động ---
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_data_from_external_api_in_background())

# --- API Endpoint để lấy dữ liệu và đưa ra dự đoán ---
@app.get("/api/taixiu")
async def get_prediction(db: Session = Depends(get_db)):
    current_time = datetime.now()
    
    # 1. Lấy thông tin phiên mới nhất từ database của bạn
    # Sử dụng kai_jiang_time để sắp xếp, an toàn hơn và chính xác hơn cho thời gian
    latest_phien = db.query(PhienTaiXiu).order_by(PhienTaiXiu.kai_jiang_time.desc()).first()
    
    phien_hien_tai_info = {}
    if latest_phien:
        phien_hien_tai_info = {
            "Expect": latest_phien.expect_string,
            "OpenCode": latest_phien.open_code,
            "KaiJiangTime": latest_phien.kai_jiang_time.isoformat(),
            "KetQuaPhien": latest_phien.ket_qua_phien,
            "TongDiem": latest_phien.tong_diem,
        }
    else:
        phien_hien_tai_info = {"ThongBao": "Chưa có dữ liệu phiên nào được xử lý hoặc lưu vào database."}

    # 2. Lấy lịch sử để đưa vào mô hình dự đoán
    # Sử dụng kai_jiang_time để sắp xếp lịch sử từ MỚI NHẤT -> CŨ NHẤT
    historical_records = db.query(PhienTaiXiu).order_by(PhienTaiXiu.kai_jiang_time.desc()).limit(HISTORY_LIMIT_FOR_ANALYSIS).all()
    historical_results_strings = [rec.ket_qua_phien for rec in historical_records if rec.ket_qua_phien]

    du_doan_phien_tiep_theo = {}

    if ml_model and historical_results_strings:
        try:
            features_for_prediction_df = extract_features(historical_results_strings)
            
            if not features_for_prediction_df.empty and list(features_for_prediction_df.columns) == FEATURE_COLUMNS:
                predicted_label = ml_model.predict(features_for_prediction_df)[0]
                
                predicted_proba = ml_model.predict_proba(features_for_prediction_df)[0]
                
                try:
                    label_index = ml_model.classes_.tolist().index(predicted_label)
                    predicted_prob_value = predicted_proba[label_index]
                    win_percentage_str = f"{predicted_prob_value * 100:.0f}% (từ mô hình ML)"
                except ValueError: 
                    print(f"Cảnh báo: Nhãn dự đoán '{predicted_label}' không tìm thấy trong classes_ của mô hình.")
                    win_percentage_str = "N/A (lỗi xác suất)"
                
                du_doan_phien_tiep_theo = {
                    "Ket_qua_du_doan": predicted_label,
                    "Ty_le_thang": win_percentage_str
                }
            else:
                print("Cảnh báo: DataFrame tính năng rỗng hoặc không khớp cột. Fallback logic thống kê cũ.")
                du_doan_phien_tiep_theo = fallback_prediction_logic(historical_results_strings)

        except Exception as e:
            print(f"Lỗi khi dự đoán bằng mô hình ML: {e}. Fallback logic thống kê cũ.")
            du_doan_phien_tiep_theo = fallback_prediction_logic(historical_results_strings)
    else:
        du_doan_phien_tiep_theo = fallback_prediction_logic(historical_results_strings)

    return {
        "status": "success",
        "current_time": current_time.isoformat(),
        "phien_hien_tai": phien_hien_tai_info,
        "Du_doan_phien_tiep_theo": du_doan_phien_tiep_theo
    }

def fallback_prediction_logic(historical_results: List[str]) -> Dict[str, str]:
    tai_count = historical_results.count("Tài")
    xiu_count = historical_results.count("Xỉu")
    total_count = len(historical_results)

    if total_count == 0:
        return {"Ket_qua_du_doan": "Không có dữ liệu", "Ty_le_thang": "0% (thống kê chung)"}

    if tai_count > xiu_count:
        predicted_outcome = "Tài"
        win_percentage = f"{(tai_count / total_count) * 100:.0f}% (thống kê chung)"
    elif xiu_count > tai_count:
        predicted_outcome = "Xỉu"
        win_percentage = f"{(xiu_count / total_count) * 100:.0f}% (thống kê chung)"
    else:
        predicted_outcome = random.choice(["Tài", "Xỉu"])
        win_percentage = "50% (thống kê chung)"

    return {"Ket_qua_du_doan": predicted_outcome, "Ty_le_thang": win_percentage}
