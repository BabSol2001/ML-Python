FROM python:3.10-slim-bookworm

WORKDIR /workspace

# تنظیم متغیرهای محیطی جهت جلوگیری از هشدارها و خطاهای DGL
ENV DGL_DISABLE_GRAPHBOLT=1
ENV DGLBACKEND=pytorch

# نصب بسته‌های ضروری سیستم‌عامل
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# نصب پکیج‌های پایتونی هماهنگ
RUN pip install --no-cache-dir "numpy<2" torch==2.2.0 torchdata==0.7.1 pandas pyyaml pydantic --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir dgl==2.1.0 -f https://data.dgl.ai/wheels/repo-ref.html

# ایجاد فایل جایگزین برای C++ GraphBolt جهت جلوگیری از خطای FileNotFoundError در اجرای اولیه
RUN cp /usr/local/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.2.0.so \
       /usr/local/lib/python3.10/site-packages/dgl/graphbolt/libgraphbolt_pytorch_2.2.2.so || true

CMD ["/bin/bash"]