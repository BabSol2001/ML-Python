# استفاده از ایمیج رسمی پایتون 3.10
FROM python:3.10-slim

# نصب وابستگی‌های سیستمی مورد نیاز C++
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt-get/lists/*

# تنظیم متغیرهای محیطی
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DGLBACKEND=pytorch

WORKDIR /workspace

# ۱. ارتقای pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ۲. محدود کردن نسخه NumPy به نسخه ۱ برای جلوگیری از تداخل با PyTorch و DGL
RUN pip install --no-cache-dir "numpy<2.0.0"

# ۳. نصب نسخه پایدار PyTorch
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# ۴. نصب DGL و سایر پکیج‌های مورد نیاز
RUN pip install --no-cache-dir \
    dgl==2.0.0 -f https://data.dgl.ai/wheels/repo.html \
    pyyaml \
    pandas \
    scipy \
    networkx \
    pymatgen \
    uvicorn \
    fastapi

# کپی کردن کدهای پروژه
COPY . /workspace/

EXPOSE 8005

# ارجاع به فایل سناریوی سیستم‌های انرژی و سیالات
CMD ["uvicorn", "fld_enrg_app:app", "--host", "0.0.0.0", "--port", "8005"]