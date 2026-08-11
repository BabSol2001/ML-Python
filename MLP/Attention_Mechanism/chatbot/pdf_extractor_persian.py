import os
import re
from pathlib import Path

def clean_and_normalize_text(text: str) -> str:
    """اصلاح کاراکترها، یکنواخت‌سازی فاصله‌ها و حفظ فرمت کدهای برنامه‌نویسی و فرمول‌ها"""
    if not text:
        return ""

    # ۱. اصلاح و استانداردسازی حروف عربی و فارسی
    translation_table = str.maketrans({
        'ي': 'ی',
        'ك': 'ک',
        'ى': 'ی',
        'ئ': 'ی',
        '١': '۱', '٢': '۲', '٣': '۳', '٤': '۴', '٥': '۵',
        '٦': '۶', '٧': '۷', '٨': '۸', '٩': '۹', '٠': '۰'
    })
    text = text.translate(translation_table)

    # ۲. حذف کاراکترهای کنترلی نامرئی و بی‌کد PDF
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # ۳. اصلاح خطوط خالی متوالی (بیش از دو خط خالی به یک خط تبدیل می‌شود)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()


def extract_tables_as_markdown(page) -> str:
    """استخراج جداول موجود در صفحه و تبدیل آن‌ها به فرمت Markdown"""
    try:
        tabs = page.find_tables()
        if not tabs.tables:
            return ""
        
        md_tables = []
        for table in tabs:
            df = table.to_pandas()
            # تبدیل جدول به ساختار متنی قابل فهم برای مدل هوش مصنوعی
            md_table = df.to_markdown(index=False)
            if md_table:
                md_tables.append(f"\n\n{md_table}\n\n")
        return "\n".join(md_tables)
    except Exception:
        return ""


def perform_ocr_on_page(page) -> str:
    """اجرای OCR در صورتی که صفحه اسکن‌شده باشد و متنی استخراج نشود"""
    try:
        import pytesseract
        from PIL import Image
        import io

        # تبدیل صفحه PDF به تصویر با کیفیت بالا
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes()))
        
        # اجرای OCR برای زبان‌های فارسی و انگلیسی
        ocr_text = pytesseract.image_to_string(img, lang='fas+eng')
        return ocr_text
    except Exception:
        return ""


def prepare_dataset(file_path: str | None = None, output_txt_path: str = "dataset.txt") -> str:
    # ۱. پرسش تعاملی آدرس فایل در صورت عدم ارسال یا خالی بودن
    if not file_path:
        file_path = input("\nلطفاً مسیر و نام فایل PDF یا TXT را وارد کنید: ").strip()
        
    # حذف کوتیشن‌های اضافی اطراف مسیر در ویندوز/لینوکس
    file_path = file_path.strip('"\'')
    path = Path(file_path)

    # بررسی وجود فایل و دریافت مجدد در صورت اشتباه بودن مسیر
    while not path.exists():
        print(f"❌ فایل یافت نشد: '{path}'")
        file_path = input("لطفاً مسیر صحیح فایل را مجدداً وارد کنید: ").strip().strip('"\'')
        path = Path(file_path)

    if path.suffix.lower() == '.pdf':
        print(f"\nفایل PDF تشخیص داده شد: '{path.name}'")
        print("در حال استخراج حرفه‌ای متن، جداول، معادلات و کدهای برنامه‌نویسی...")

        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            raise ImportError(
                "کتابخانه PyMuPDF نصب نیست! لطفاً دستور زیر را اجرا کنید:\n"
                "pip install pymupdf pandas tabulate"
            )

        extracted_pages = []

        with pymupdf.open(str(path)) as doc:
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # ۱. دریافت خروجی خام و تبدیل صریح نوع داده به str برای رفع خطای Pylance
                raw_text = page.get_text("text")
                
                if isinstance(raw_text, list):
                    text = "\n".join(str(item) for item in raw_text)
                elif isinstance(raw_text, dict):
                    text = str(raw_text)
                else:
                    text = str(raw_text or "")

                # ۲. استخراج جداول (اطمینان از رشته بودن خروجی)
                tables_md: str = extract_tables_as_markdown(page)
                
                # ۳. اگر متن و جدول هر دو خالی بودند، اجرای OCR
                if not text.strip() and not tables_md.strip():
                    text = perform_ocr_on_page(page)

                combined_page_content = f"{text}\n{tables_md}".strip()

                if combined_page_content:
                    cleaned_page = clean_and_normalize_text(combined_page_content)
                    if cleaned_page:
                        extracted_pages.append(f"--- صفحه {page_num + 1} ---\n{cleaned_page}")

        full_text = "\n\n".join(extracted_pages)

        if not full_text.strip():
            raise ValueError("هیچ محتوایی از فایل PDF استخراج نشد!")

        # ذخیره نهایی دیتاست به صورت UTF-8
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        print(f"✅ استخراج حرفه‌ای با موفقیت انجام شد. دیتاست در '{output_txt_path}' ذخیره گردید.\n")
        return output_txt_path

    elif path.suffix.lower() == '.txt':
        return str(path)
    else:
        raise ValueError("فرمت فایل پشتیبانی نمی‌شود. فقط فایل‌های PDF و TXT مجاز هستند.")


if __name__ == "__main__":
    # امکان اجرای مستقیم فایل جهت تست
    prepare_dataset()