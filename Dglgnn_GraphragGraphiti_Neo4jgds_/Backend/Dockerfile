# استفاده از ایمیج رسمی پایتون 3.10
FROM python:3.10-slim

# نصب وابستگی‌های سیستمی مورد نیاز C++ و ابزارهای شبکه
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    curl \
    git \
    netcat-openbsd \
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

# ۳. نصب نسخه پایدار PyTorch (نسخه CPU)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# ۴. نصب DGL، Neo4j Driver، FastAPI و سایر پکیج‌های مورد نیاز
RUN pip install --no-cache-dir \
    dgl==2.0.0 -f https://data.dgl.ai/wheels/repo.html \
    neo4j \
    pyyaml \
    pandas \
    scipy \
    networkx \
    uvicorn \
    fastapi \
    pydantic \
    requests

# کپی کردن کدهای پروژه به کانتینر
COPY . /workspace/

# باز کردن پورت سرویس اصلی برنامه
EXPOSE 8008

# اجرا و ارجاع نهایی به نقطه ورود اصلی سیستم (prdctv_mtn_grp_neo_app.py)
CMD ["uvicorn", "prdctv_mtn_grp_neo_app:app", "--host", "0.0.0.0", "--port", "8008"]