FROM python:3.10-slim-bookworm

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    networkx \
    python-louvain \
    torch \
    fastapi \
    uvicorn \
    pydantic

# 🔹 اضافه کردن فایل‌های پایتون به کانتینر
COPY . /workspace

EXPOSE 8002

CMD ["sh", "-c", "python pipeline_graphrag.py && uvicorn app_graphrag:app --host 0.0.0.0 --port 8002"]