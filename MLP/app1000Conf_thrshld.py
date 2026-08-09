import io
import torch
from torchvision import models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

app = FastAPI(
    title="Hybrid ResNet18 Classification API",
    description="سرویس تشخیص ۱۰۰۰ کلاس ImageNet به همراه آستانه اطمینان برای شناسایی ورودی‌های نامعتبر"
)

# -------------------------------------------------------------
# ۱. تنظیمات و آستانه اطمینان (Confidence Threshold)
# -------------------------------------------------------------
CONFIDENCE_THRESHOLD = 60.0  # حداقل درصد اطمینان برای پذیرش پاسخ (مثلاً ۶۰٪)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------
# ۲. بارگذاری مدل ۱۰۰۰ کلاسی ImageNet و تبدیل‌های استاندارد
# -------------------------------------------------------------
weights = models.ResNet18_Weights.DEFAULT
categories = weights.meta["categories"]  # اسامی ۱۰۰۰ کلاس ImageNet
transform = weights.transforms()          # تبدیل‌های استاندارد متناسب با مدل

try:
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()
    print("✅ مدل ResNet-18 ImageNet با موفقیت بارگذاری شد.")
except Exception as e:
    print(f"❌ خطا در بارگذاری مدل: {e}")
    model = None

# -------------------------------------------------------------
# ۳. Endpoint پیش‌بینی ترکیب‌شده (/predict)
# -------------------------------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="مدل هنوز در سرور بارگذاری نشده است.")
    
    # اعتبار سنجی نوع فایل
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="فایل ارسالی باید یک تصویر باشد.")

    try:
        # ۱. خواندن و تبدیل بایت‌های تصویر
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ۲. تبدیل تصویر به تانسور پایتورچ [1, 3, H, W]
        image_tensor = transform(image)
        if not isinstance(image_tensor, torch.Tensor):
            raise HTTPException(status_code=500, detail="خطا در تبدیل تصویر به تانسور.")
            
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        # ۳. استنتاج (Inference)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
        class_index = int(predicted_idx.item())
        predicted_class = categories[class_index]
        confidence_percentage = round(float(confidence.item()) * 100, 2)
        
        # 🎯 بررسی معیار آستانه اطمینان (Confidence Threshold)
        if confidence_percentage < CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "filename": file.filename,
                "is_confident": False,
                "prediction": {
                    "class": "Unknown / Low Confidence Input",
                    "confidence": f"{confidence_percentage}%",
                    "top_guess": predicted_class,
                    "message": f"اطمینان مدل کمتر از حد مجاز ({CONFIDENCE_THRESHOLD}%) است."
                }
            }
        
        # خروجی موفقیت‌آمیز برای تصاویر شناساپی‌شده
        return {
            "success": True,
            "filename": file.filename,
            "is_confident": True,
            "prediction": {
                "class": predicted_class,
                "confidence": f"{confidence_percentage}%"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش تصویر: {str(e)}")

# -------------------------------------------------------------
# ۴. اجرا با Uvicorn
# -------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)