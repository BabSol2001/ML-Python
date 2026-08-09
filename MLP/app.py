import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

app = FastAPI(
    title="ResNet18 CIFAR-10 Classification API",
    description="سرویس تشخیص ۱۰ کلاس تصویر با استفاده از مدل ResNet-18 در PyTorch"
)

# -------------------------------------------------------------
# ۱. تنظیم سخت‌افزار و کلاس‌های CIFAR-10
# -------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -------------------------------------------------------------
# ۲. آماده‌سازی تبدیل‌های تصویر (دقیقاً مشابه زمان آموزش)
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# ۳. ساخت معماری و بارگذاری وزن‌های ذخیره‌شده
# -------------------------------------------------------------
def load_resnet_model(weights_path: str = "resnet18_cifar10.pth") -> nn.Module:
    model = models.resnet18(weights=None)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

try:
    model = load_resnet_model()
    print("✅ مدل ResNet-18 با موفقیت در FastAPI بارگذاری شد.")
except Exception as e:
    print(f"❌ خطا در بارگذاری مدل: {e}")
    model = None

# -------------------------------------------------------------
# ۴. ساخت Endpoint پیش‌بینی (/predict)
# -------------------------------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="مدل هنوز در سرور بارگذاری نشده است.")
    
    # اصلاح ۱: چک کردن Optional بودن content_type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="فایل ارسالی باید یک تصویر باشد.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # اصلاح ۲: کست کردن شفاف خروجی transform به Tensor
        image_tensor = transform(image)
        if not isinstance(image_tensor, torch.Tensor):
            raise HTTPException(status_code=500, detail="خطا در تبدیل تصویر به تانسور.")
            
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        # ۳. استنتاج (Inference)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
        # اصلاح ۳: تبدیل صریح خروجی پایتورچ به int برای ایندکس لیست
        class_index = int(predicted_idx.item())
        predicted_class = CLASSES[class_index]
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)