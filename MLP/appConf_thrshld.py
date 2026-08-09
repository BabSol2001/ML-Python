import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

app = FastAPI(
    title="ResNet18 CIFAR-10 API with Threshold",
    description="سرویس تشخیص ۱۰ کلاس CIFAR-10 به همراه فیلتر تصاویر خارج از محدوده"
)

# -------------------------------------------------------------
# ۱. تنظیمات و آستانه اطمینان (Confidence Threshold)
# -------------------------------------------------------------
CONFIDENCE_THRESHOLD = 65.0  # حداقل درصد اطمینان برای پذیرش پاسخ (مثلاً ۶۵٪)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# ۲. بارگذاری مدل
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
    print("✅ مدل با موفقیت بارگذاری شد.")
except Exception as e:
    print(f"❌ خطا در بارگذاری مدل: {e}")
    model = None

# -------------------------------------------------------------
# ۳. Endpoint پیش‌بینی با فیلتر Confidence Threshold
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
        predicted_class = CLASSES[class_index]
        confidence_percentage = round(float(confidence.item()) * 100, 2)
        
        # 🎯 اعمال آستانه اطمینان (Confidence Threshold Check)
        if confidence_percentage < CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "filename": file.filename,
                "is_known_class": False,
                "prediction": {
                    "class": "Unknown / Out of CIFAR-10 Domain",
                    "confidence": f"{confidence_percentage}%",
                    "top_guess": predicted_class,
                    "message": f"میزان اطمینان مدل کمتر از حد مجاز ({CONFIDENCE_THRESHOLD}%) است."
                }
            }
        
        # خروجی معمولی برای تصاویر معتبر
        return {
            "success": True,
            "filename": file.filename,
            "is_known_class": True,
            "prediction": {
                "class": predicted_class,
                "confidence": f"{confidence_percentage}%"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش تصویر: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)