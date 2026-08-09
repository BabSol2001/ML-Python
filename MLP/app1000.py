import io
import torch
from torchvision import models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

app = FastAPI(
    title="ResNet18 ImageNet Classification API",
    description="سرویس تشخیص ۱۰۰۰ کلاس تصویر با استفاده از مدل پیش‌آموخته ResNet-18"
)

# -------------------------------------------------------------
# ۱. تنظیم سخت‌افزار
# -------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------
# ۲. دریافت وزن‌های ImageNet، تبدیل‌ها و اسامی ۱۰۰۰ کلاس
# -------------------------------------------------------------
weights = models.ResNet18_Weights.DEFAULT
categories = weights.meta["categories"]  # لیست ۱۰۰۰ کلاس رسمی ImageNet
transform = weights.transforms()          # تبدیل‌های خودکار استاندارد متناسب با این وزن‌ها

# -------------------------------------------------------------
# ۳. بارگذاری مدل بدون تعویض لایه آخر
# -------------------------------------------------------------
try:
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()
    print("✅ مدل ResNet-18 با ۱۰۰۰ کلاس ImageNet با موفقیت بارگذاری شد.")
except Exception as e:
    print(f"❌ خطا در بارگذاری مدل: {e}")
    model = None

# -------------------------------------------------------------
# ۴. Endpoint پیش‌بینی (/predict)
# -------------------------------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="مدل هنوز در سرور بارگذاری نشده است.")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="فایل ارسالی باید یک تصویر باشد.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # اعمال تبدیل و آماده‌سازی تانسور ورودی
        image_tensor = transform(image)
        if not isinstance(image_tensor, torch.Tensor):
            raise HTTPException(status_code=500, detail="خطا در تبدیل تصویر به تانسور.")
            
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        # استنتاج
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
        class_index = int(predicted_idx.item())
        predicted_class = categories[class_index]
        confidence_percentage = round(float(confidence.item()) * 100, 2)
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": {
                "class": predicted_class,
                "confidence": f"{confidence_percentage}%"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش تصویر: {str(e)}")

# -------------------------------------------------------------
# ۵. اجرا با Uvicorn
# -------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)