import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please set it for Render deployment.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PhienTaiXiu(Base):
    __tablename__ = "phien_tai_xiu"

    id = Column(Integer, primary_key=True, index=True)
    expect_string = Column(String, unique=True, index=True, nullable=False) # ID của phiên
    open_code = Column(String) # Mã mở thưởng (ví dụ: '1,2,3')
    kai_jiang_time = Column(DateTime) # Thời gian mở thưởng
    ket_qua_phien = Column(String) # Kết quả của phiên (Tài/Xỉu)
    tong_diem = Column(Integer) # Tổng điểm của phiên
    created_at = Column(DateTime, default=datetime.utcnow) # Thời gian record vào DB

    def __repr__(self):
        return f"<PhienTaiXiu(expect_string={self.expect_string}, ket_qua_phien={self.ket_qua_phien}, tong_diem={self.tong_diem})>"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
