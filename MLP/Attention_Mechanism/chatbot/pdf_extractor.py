import os
from pathlib import Path

def prepare_dataset(file_path: str, output_txt_path: str = "dataset.txt") -> str:
    """
    بررسی پسوند فایل:
    - اگر PDF باشد، متن آن را استخراج کرده و در یک فایل txt ذخیره می‌کند.
    - اگر TXT باشد، همان را آماده استفاده می‌کند.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"فایل ورودی '{file_path}' یافت نشد!")

    # اگر فایل ورودی PDF باشد
    if path.suffix.lower() == '.pdf':
        print(f"فایل PDF تشخیص داده شد: '{file_path}'")
        print("در حال استخراج متن از فایل PDF...")
        
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "کتابخانه 'pypdf' نصب نیست! لطفاً آن را با دستور زیر نصب کنید:\n"
                "pip install pypdf"
            )

        reader = PdfReader(str(path))
        extracted_text = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_text.append(text)
                
        full_text = "\n".join(extracted_text)
        
        if not full_text.strip():
            raise ValueError("هیچ متنی از فایل PDF استخراج نشد! ممکن است فایل اسکن‌شده یا تصویری باشد.")

        # ذخیره متن استخراج‌شده در فایل txt
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        print(f"استخراج با موفقیت انجام شد. متن در '{output_txt_path}' ذخیره گردید.\n")
        return output_txt_path

    # اگر فایل از قبل TXT باشد
    elif path.suffix.lower() == '.txt':
        print(f"فایل متنی TXT تشخیص داده شد: '{file_path}'\n")
        return file_path
    
    else:
        raise ValueError(f"فرمت فایل '{path.suffix}' پشتیبانی نمی‌شود. لطفاً فایل PDF یا TXT وارد کنید.")