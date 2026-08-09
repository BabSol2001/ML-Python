import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ۱. تعریف ساختار ورودی داده‌های خام با Pydantic
class UserDataInput(BaseModel):
    age: Optional[float] = Field(None, example=35.0, description="سن کاربر (می‌تواند نال باشد)")
    income: Optional[float] = Field(None, example=65000.0, description="درآمد سالانه")
    education: Optional[str] = Field(None, example="Bachelor", description="مدرک تحصیلی")
    city: Optional[str] = Field(None, example="Tehran", description="شهر محل سکونت")

# ۲. ساخت برنامه FastAPI
app = FastAPI(
    title="ML Pipeline Prediction API",
    description="API برای دریافت داده‌های خام و خروجی مدل XGBoost",
    version="1.0.0"
)

# ۳. بارگذاری پایپ‌لاین در زمان بالا آمدن برنامه‌
PIPELINE_PATH = "pipeline_model.joblib"
try:
    pipeline = joblib.load(PIPELINE_PATH)
    print("✅ پایپ‌لاین با موفقیت بارگذاری شد.")
except Exception as e:
    print(f"❌ خطای بارگذاری مدل: {e}")
    pipeline = None

# ۴. تعریف مسیر تست سلامت سرویس (Health Check)
@app.get("/")
def home():
    return {"status": "online", "message": "سرویس پیش‌بینی فعال است."}

# ۵. تعریف Endpoint اصلی پیش‌بینی
@app.post("/predict")
def predict(data: UserDataInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="مدل روی سرور بارگذاری نشده است.")
    
    try:
        # تبدیل ورودی JSON به یک DataFrame تک‌سطری پانداس
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # پیش‌بینی کلاس و احتمال خروجی
        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])
        
        # بازگرداندن پاسخ JSON
        return {
            "prediction_class": prediction,
            "probability_positive_class": round(probability, 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطا در پردازش ورودی: {str(e)}")