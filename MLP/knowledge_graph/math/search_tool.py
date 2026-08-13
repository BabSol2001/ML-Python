import re
import sympy as sp
from ddgs import DDGS


class SmartEducationalSearchEngine:
    """موتور جستجوی هوشمند آموزشی با فیلتر محتوا، رتبه‌بندی و بانک پشتیبان آفلاین"""

    def __init__(self):
        self.ddgs = DDGS()

        # کلمات ممنوعه جهت فیلتر کردن اخبار، ورزش و صفحات تجاری
        self.blacklist = [
            'real madrid',
            'football',
            'soccer',
            'news',
            'sport',
            'betting',
            'crypto',
            'shop',
            'price',
            'ticket',
            'casino',
            'deal',
            'discount',
            'sale',
        ]

        # نگاشت مفاهیم آموزشی به کلیدواژه‌های تخصصی انگلیسی جهت جستجوی دقیق
        self.concept_query_map = {
            'separation_of_variables': {
                'queries': [
                    'separation of variables physical intuition heat equation',
                    'separation of variables real life analogy PDE',
                ],
                'must_contain': [
                    'variable',
                    'product',
                    'function',
                    'pde',
                    'wave',
                    'heat',
                    'separate',
                ],
            },
            'spatial_boundary': {
                'queries': [
                    'heat equation boundary conditions physical meaning',
                    'sine wave temperature distribution rod fixed end',
                ],
                'must_contain': [
                    'boundary',
                    'zero',
                    'fixed',
                    'sine',
                    'temperature',
                    'ends',
                    'rod',
                ],
            },
            'exponential_decay': {
                'queries': [
                    'exponential decay heat transfer intuitive example',
                    'thermal relaxation time constant physics analogy',
                ],
                'must_contain': [
                    'decay',
                    'exponential',
                    'cooling',
                    'temperature',
                    'time',
                    'rate',
                ],
            },
            'initial_condition': {
                'queries': [
                    'heat equation initial condition Fourier series physical meaning',
                    'initial temperature profile rod physics intuition',
                ],
                'must_contain': [
                    'initial',
                    'profile',
                    'mode',
                    'fourier',
                    'shape',
                    'temperature',
                ],
            },
        }

        # بانک اطلاعاتی پشتیبان در صورت عدم دریافت پاسخ باکیفیت از وب
        self.fallback_analogies = {
            'separation_of_variables': (
                '💡 [مثال ملموس]: مثل بررسی ارتعاش سیم گیتار است؛ شکل کلی سیم (مکان) و سرعت لرزش آن در طول زمان '
                'را دو موضوع مستقل فرض کرده و در هم ضرب می‌کنیم.'
            ),
            'spatial_boundary': (
                '💡 [مثال ملموس]: مثل نگه داشتن دو سر یک میله فلزی در قالب یخ است. دما در دو انتها همواره صفر می‌ماند '
                'و دمای وسط میله به شکل یک کمان سینوسی توزیع می‌شود.'
            ),
            'exponential_decay': (
                '💡 [مثال ملموس]: مثل داغ کردن یک قاشق و رها کردن آن روی میز است. افت دما در ابتدا بسیار سریع است '
                'و با نزدیک شدن به دمای محیط، سرعت سرد شدن کندتر می‌شود (افت نمایی).'
            ),
            'initial_condition': (
                '💡 [مثال ملموس]: مثل اثر انگشت حرارتی اولیه است؛ شکل توزیع دما در لحظه شروع (t=0) مشخص می‌کند '
                'که کدام حالت‌های نوسانی (Harmonics) در ادامه باقی می‌مانند.'
            ),
        }

    def _clean_text(self, text: str) -> str:
        """تصفیه متن و حذف کدهای HTML و فاصله‌های زاید"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _score_relevance(self, text: str, concept_key: str) -> int:
        """محاسبه امتیاز ارتباط متن با موضوع فیزیکی/ریاضی"""
        text_lower = text.lower()

        # ۱. اگر شامل کلمات لیست سیاه باشد، امتیاز صفر می‌شود
        if any(bad_word in text_lower for bad_word in self.blacklist):
            return -100

        score = 0
        concept_info = self.concept_query_map.get(concept_key, {})
        must_keywords = concept_info.get('must_contain', [])

        # ۲. محاسبه امتیاز بر اساس وجود کلیدواژه‌های تخصصی
        for kw in must_keywords:
            if kw in text_lower:
                score += 15

        # کلمات عمومی تقویت‌کننده مرتبط با فیزیک و ریاضی
        general_physics_words = [
            'temperature',
            'heat',
            'diffusivity',
            'pde',
            'equation',
            'solution',
            'decay',
            'cooling',
        ]
        for word in general_physics_words:
            if word in text_lower:
                score += 5

        return score

    def search_educational_analogy(
        self, concept_key: str, max_results: int = 3
    ) -> str:
        """جستجوی هوشمند در وب، پالایش، نمره‌دهی و ارائه بهترین مثال آموزشی"""
        concept_info = self.concept_query_map.get(concept_key)

        if not concept_info:
            return self.fallback_analogies.get(
                concept_key, "مثال متناسب یافت نشد."
            )

        queries = concept_info['queries']
        candidate_snippets = []

        # اجرا روی پرس‌وجوهای اختصاصی
        for query in queries:
            try:
                raw_results = list(self.ddgs.text(query, max_results=max_results))
                for res in raw_results:
                    body = self._clean_text(res.get('body', ''))
                    score = self._score_relevance(body, concept_key)

                    if score > 20:  # حداقل آستانه کیفیت برای پذیرش متن
                        candidate_snippets.append((score, body))
            except Exception:
                continue

        # اگر نتایج باکیفیتی پیدا شد، بهترین آن‌ها بر اساس امتیاز انتخاب می‌شود
        if candidate_snippets:
            candidate_snippets.sort(key=lambda x: x[0], reverse=True)
            best_snippet = candidate_snippets[0][1]
            return f"💡 [نکته و مثال استخراج شده از وب]:\n\"{best_snippet}\""

        # استفاده از پشتیبان آفلاین در صورت عدم یافتن نتیجه مناسب
        return self.fallback_analogies.get(
            concept_key, "استفاده از راهنمایی‌های پایه."
        )


class SocraticMathTutor:

    def __init__(self):
        # ابزار جدید و ارتقایافته جستجوی هوشمند
        self.search_tool = SmartEducationalSearchEngine()

        self.x, self.t, self.L, self.alpha, self.n = sp.symbols(
            'x t L alpha n', positive=True, real=True
        )
        self.pi = sp.pi

        self.symbols_dict = {
            'x': self.x,
            't': self.t,
            'L': self.L,
            'alpha': self.alpha,
            'n': self.n,
            'pi': self.pi,
            'sin': sp.sin,
            'cos': sp.cos,
            'exp': sp.exp,
        }

    def evaluate_sympy_expression(self, user_input: str, target_expr) -> bool:
        """سنجش صحت ریاضی ورودی دانش‌آموز با SymPy"""
        try:
            clean_input = user_input.replace('^', '**')
            user_expr = sp.sympify(clean_input, locals=self.symbols_dict)
            diff = sp.simplify(user_expr - target_expr)
            return diff == 0
        except Exception:
            return False

    def is_math_expression(self, user_input: str) -> bool:
        """تشخیص اینکه آیا ورودی شامل نمادهای فرمولی است یا متن کلامی"""
        math_chars = ['*', '/', '+', '-', '**', '^', '(', ')']
        return any(char in user_input for char in math_chars)

    def evaluate_verbal_concept(self, user_input: str, keywords: list) -> bool:
        """سنجش صحت توضیحات کلامی بر اساس کلیدواژه‌ها"""
        user_input_lower = user_input.lower()
        return any(kw in user_input_lower for kw in keywords)

    def ask_socratic_step(
        self,
        title: str,
        question: str,
        target_expr,
        hints: list,
        concept_key: str,
        concept_keywords=None,
        validation_type='sympy',
    ):
        """اجرای گام تعاملی با ارزیابی هوشمند، تشخیص غلط‌های متداول و سرچ خودکار"""
        print(
            f'\n============================================================'
        )
        print(f'📌 {title}')
        print(
            f'============================================================'
        )
        print(question)

        attempt = 0
        while True:
            user_input = input('\n✏️ پاسخ شما: ').strip()

            if not user_input:
                print('لطفاً پاسخ یا حدس خود را وارد کنید.')
                continue

            # فاز ۱: ارزیابی ریاضی/کلیدواژه‌ای مستقیم
            is_correct = False
            if validation_type == 'sympy':
                is_correct = self.evaluate_sympy_expression(
                    user_input, target_expr
                )
            elif validation_type == 'keyword':
                is_correct = any(
                    kw in user_input.lower() for kw in target_expr
                )

            if is_correct:
                print(
                    '\n✅ آفرین! پاسخ شما کاملاً درست است. استدلال ریاضی شیوایی داشتید.'
                )
                break

            # فاز ۱.۴: تشخیص اشتباه T بزرگ (دما) به جای t کوچک (زمان)
            if validation_type == 'sympy' and 'T' in user_input:
                print('\n💡 نکته متغیرها: شما از حرف T بزرگ استفاده کرده‌اید!')
                print(
                    'در این معادله T نماد دما و t نماد زمان است. لطفاً از t کوچک استفاده کنید.'
                )
                continue

            # فاز ۱.۵: بررسی هوشمند متغیر اضافی n (مانند n در گام ۴)
            if (
                validation_type == 'sympy'
                and 'n' in user_input
                and 'n' not in str(target_expr)
            ):
                print(
                    '\n💡 نکته دقیق: شما متغیر n را در فرمول باقی گذاشته‌اید!'
                )
                print(
                    'با توجه به شرط اولیه، مقدار n برابر 1 است. لطفاً n را با 1 جایگزین کنید.'
                )
                continue

            # فاز ۲: پشتیبانی از توضیحات کلامی
            if (
                validation_type == 'sympy'
                and concept_keywords
                and not self.is_math_expression(user_input)
            ):
                if self.evaluate_verbal_concept(user_input, concept_keywords):
                    print(
                        '\n💡 مفهوم منطقی کلام شما کاملاً درست است! آفرین که استدلال فیزیکی/ریاضی مسئله را متوجه شدید.'
                    )
                    print('حالا همین مفهومی را که گفتی به‌صورت فرمول دقیق بنویس.')
                    continue

            # فاز ۳: ارائه راهنمایی‌های تعاملی و فراخوانی موتور جستجوی هوشمند
            attempt += 1
            print(
                '\nایده خوبیه، اما هنوز به فرم دقیق نرسیدیم. بیا با هم مرور کنیم:'
            )

            if attempt == 1:
                print(f'💡 راهنمایی ۱ (اشاره مفهومی): {hints[0]}')
            elif attempt == 2:
                print(f'💡 راهنمایی ۲ (ساختار ریاضی): {hints[1]}')
            elif attempt == 3:
                print(f'💡 راهنمایی ۳ (کمک مستقیم): {hints[2]}')

                # فراخوانی موتور جستجوی هوشمند با کلید مفهومی استاندارد
                web_example = self.search_tool.search_educational_analogy(
                    concept_key
                )
                print(f'\n🔍 [تحلیل هوشمند موتور سرچ]:\n{web_example}')
            else:
                print(f'✨ پاسخ این مرحله: {target_expr}')
                print(
                    'اصلاً نگران نباش! هدف اصلی درک روند حل است. بریم سراغ گام بعدی.'
                )
                break


# ---------------------------------------------------------
# نمونه تست و اجرای برنامه
# ---------------------------------------------------------
if __name__ == '__main__':
    tutor = SocraticMathTutor()

    print(
        '============================================================'
    )
    print('    🎓 مدرس هوشمند سقراطی: معادله انتقال حرارت ۱بعدی')
    print(
        '============================================================'
    )

    # گام ۱: تفکیک متغیرها
    tutor.ask_socratic_step(
        title='گام ۱: ایده اصلی حل',
        question='برای حل این معادله، فرض می‌کنیم پاسخ حاصل‌ضرب دو تابع جداگانه T(x,t) = X(x) * T_time(t) است.\nاسم این تکنیک چیست؟',
        target_expr=['تفکیک متغیرها', 'separation of variables'],
        hints=[
            'به مجزا کردن متغیرهای x و t اشاره دارد.',
            'عبارت شامل کلمه "تفکیک" است.',
            'نام این روش "تفکیک متغیرها" است.',
        ],
        concept_key='separation_of_variables',
        validation_type='keyword',
    )

    # گام ۳: بخش زمانی
    target_time = sp.exp(-tutor.alpha * (tutor.pi / tutor.L) ** 2 * tutor.t)
    tutor.ask_socratic_step(
        title='گام ۳: بخش زمانی پاسخ T_time(t)',
        question='معادله بخش زمانی به صورت dT/dt + alpha*(n*pi/L)^2 * T = 0 است.\nپاسخ این معادله نمایی را بنویسید (از exp استفاده کنید):',
        target_expr=target_time,
        hints=[
            'جواب عمومی به صورت exp(-k*t) است.',
            'k برابر alpha*(pi/L)^2 است (با فرض n=1).',
            'فرمول کامل exp(-alpha*(pi/L)^2*t) می‌باشد.',
        ],
        concept_key='exponential_decay',
        concept_keywords=['نمایی', 'توان', 'exp', 'کاهشی'],
        validation_type='sympy',
    )