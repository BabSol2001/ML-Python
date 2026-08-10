import os
import re
from pathlib import Path

def clean_extracted_text(text: str) -> str:
    """
    پاک‌سازی نویزهای استخراج PDF و حفظ حروف فارسی، انگلیسی، 
    علائم برنامه‌نویسی پایتون و نمادهای ریاضی پایه.
    """
    # ۱. حذف آیکون‌ها، شکلک‌ها و کاراکترهای غیرمعمول با رنج یونی‌کد (ایموجی‌ها و سمبل‌ها)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    
    # ۲. استانداردسازی خطوط جدید (جلوگیری از خطوط خالی متعدد)
    text = re.sub(r'\n+', '\n', text)
    
    # ۳. حذف فاصله‌های متوالی بدون دستکاری تب‌ها و اینترهای کد
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def prepare_dataset(file_path: str, output_txt_path: str = "dataset.txt") -> str:
    """
    بررسی پسوند فایل و استخراج استاندارد متن PDF یا TXT
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"فایل ورودی '{file_path}' یافت نشد!")

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
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
                
        full_text = "\n".join(extracted_text)
        
        if not full_text.strip():
            raise ValueError("هیچ متنی از فایل PDF استخراج نشد! ممکن است فایل اسکن‌شده یا تصویری باشد.")

        # تمیزکاری متن ترکیبی (فارسی، انگلیسی، کد، ریاضی)
        cleaned_text = clean_extracted_text(full_text)

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            
        print(f"استخراج با موفقیت انجام شد. متن در '{output_txt_path}' ذخیره گردید.\n")
        return output_txt_path

    elif path.suffix.lower() == '.txt':
        print(f"فایل متنی TXT تشخیص داده شد: '{file_path}'\n")
        return file_path
    
    else:
        raise ValueError(f"فرمت فایل '{path.suffix}' پشتیبانی نمی‌شود. لطفاً فایل PDF یا TXT وارد کنید.")