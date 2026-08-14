import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch

from integration_test import exact_solution
# وارد کردن ابزارهای وابسته به پروژه
from pinn import train_pinn
from search_tool import SmartEducationalSearchEngine


class SocraticMathTutor:
    """آموزگار تعاملی سقراطی با ارزیابی جبر نمادین، درک کلامی و اتصال به PINN"""

    def __init__(self):
        # تعریف متغیرهای نمادین ریاضی در SymPy
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

        # ابزار موتور جستجوی هوشمند آموزشی
        self.search_tool = SmartEducationalSearchEngine()

    def evaluate_sympy_expression(
        self, user_input: str, target_expr: sp.Expr
    ) -> bool:
        """سنجش صحت ریاضی ورودی دانش‌آموز با SymPy"""
        try:
            # تبدیل علامت توان معمولی به توان پایتونی
            clean_input = user_input.replace('^', '**')
            user_expr = sp.sympify(clean_input, locals=self.symbols_dict)
            diff = sp.simplify(user_expr - target_expr)
            return diff == 0
        except Exception:
            return False

    def evaluate_verbal_concept(
        self, user_input: str, keywords: list[str]
    ) -> bool:
        """سنجش صحت توضیحات کلامی دانش‌آموز بر اساس کلیدواژه‌ها"""
        user_input_lower = user_input.lower()
        return any(kw in user_input_lower for kw in keywords)

    def is_math_expression(self, user_input: str) -> bool:
        """تشخیص اینکه ورودی، فرمول ریاضی است یا متن کلامی"""
        math_chars = ['*', '/', '+', '-', '**', '^', '(', ')']
        return any(char in user_input for char in math_chars)

    def _trigger_educational_search(self, concept_key: str):
        """فراخوانی موتور جستجوی زنده و ارائه تحلیل ملموس آموزشی"""
        print('\n🔍 [در حال جستجو در وب و سنتز آموزش انسانی...]')

        # فراخوانی متد اصلی سنتز پویا
        if hasattr(self.search_tool, 'get_dynamic_explanation'):
            explanation = self.search_tool.get_dynamic_explanation(concept_key)
        else:
            explanation = 'تحلیل آموزشی در دسترس نیست.'

        words_count = len(explanation.split())
        print(
            '------------------------------------------------------------'
        )
        print(
            f'🌐 [عصاره جستجو و تحلیل زنده وب | طول متن: {words_count} واژه]:'
        )
        print(
            '------------------------------------------------------------'
        )
        print(explanation)
        print(
            '------------------------------------------------------------\n'
        )

    def ask_socratic_step(
        self,
        title: str,
        question: str,
        target_expr,
        hints: list[str],
        concept_key: str,
        concept_keywords: list[str] = [],
        validation_type: str = 'sympy',
    ):
        """اجرای یک گام تعاملی هوشمند با پشتیبانی از ارزیابی‌های چندلایه و سرچ هوشمند"""
        if concept_keywords is None:
            concept_keywords = []

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

            # فاز ۱: ارزیابی ریاضی / کلیدواژه‌ای مستقیم
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
                self._trigger_educational_search(concept_key)
                break

            # فاز ۱.۴: تشخیص هوشمند خطای T بزرگ (دما) به جای t کوچک (زمان)
            if validation_type == 'sympy' and 'T' in user_input:
                print('\n💡 نکته متغیرها: شما از حرف T بزرگ استفاده کرده‌اید!')
                print(
                    'در این معادله T نماد دما و t نماد زمان است. لطفاً از t کوچک استفاده کنید.'
                )
                continue

            # فاز ۱.۵: تشخیص هوشمند باقی ماندن متغیر اضافی n (مثل گام ۴)
            if (
                validation_type == 'sympy'
                and 'n' in user_input
                and 'n' not in str(target_expr)
            ):
                print(
                    '\n💡 نکته دقیق: شما متغیر n را در فرمول باقی گذاشته‌اید!'
                )
                print(
                    'با توجه به شرط اولیه، مقدار n برابر 1 است. لطفاً n را با 1 جایگزین کرده و دوباره بنویسید.'
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

            # فاز ۳: ارائه راهنمایی‌ها و فراخوانی موتور جستجوی هوشمند
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
                self._trigger_educational_search(concept_key)
            else:
                print(f'✨ پاسخ این مرحله: {target_expr}')
                print(
                    'اصلاً نگران نباش! هدف اصلی درک روند حل است. بریم سراغ گام بعدی.'
                )
                self._trigger_educational_search(concept_key)
                break

    def run_pinn_verification_demo(self, user_final_expr):
        """اتصال به فایل‌های pinn.py و integration_test.py جهت شبیه‌سازی هوش مصنوعی"""
        print(
            '\n============================================================'
        )
        print('🤖 ورود به لایه هوش مصنوعی و شبیه‌سازی (PINN Verification)')
        print(
            '============================================================'
        )
        print(
            'حالا که شما فرمول تحلیلی را استخراج کردید، می‌خواهیم شبکه عصبی فیزیک‌محوری را که'
        )
        print(
            'در فایل pinn.py تعریف کرده‌ایم آموزش دهیم و دقت فرمول شما را با هوش مصنوعی بسنجیم!\n'
        )

        confirm = input(
            'آیا مایلید شبکه عصبی را آموزش داده و نمودار مقایسه را تولید کنیم؟ (y/n): '
        )
        if confirm.lower() == 'y':
            print('\n[در حال آموزش شبکه عصبی PINN... لطفاً شکیبا باشید]')
            model = train_pinn()

            x_test = np.linspace(0, 1.0, 100).reshape(-1, 1)
            t_test = np.full_like(x_test, 0.1)

            x_tensor = torch.tensor(x_test, dtype=torch.float32)
            t_tensor = torch.tensor(t_test, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                T_pinn = model(x_tensor, t_tensor).numpy()

            T_exact = exact_solution(x_test, 0.1, alpha=0.01, L=1.0)
            mae_error = np.mean(np.abs(T_pinn - T_exact))

            plt.figure(figsize=(8, 5))
            plt.plot(
                x_test,
                T_exact,
                'r-',
                label='Exact Analytical (Your Solution)',
                linewidth=2,
            )
            plt.plot(
                x_test,
                T_pinn,
                'b--',
                label='PINN Neural Network',
                linewidth=2,
            )
            plt.title(
                f'Socratic Math AI: Exact vs Neural Net (MAE: {mae_error:.6f})'
            )
            plt.xlabel('x (Position)')
            plt.ylabel('T (Temperature)')
            plt.legend()
            plt.grid(True)

            output_file = 'tutor_pinn_result.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(
                f'\n🎉 فوق‌العاده است! فرمولی که به دست آوردید با خطای MAE = {mae_error:.6f} توسط شبکه عصبی تایید شد.'
            )
            print(f"نمودار مقایسه در فایل '{output_file}' ذخیره شد.")

    def start(self):
        print("""
============================================================
    🎓 مدرس هوشمند سقراطی: معادله انتقال حرارت ۱بعدی
============================================================
سلام! من همراه آموزش ریاضی شما هستم. قرار است گام‌به‌گام معادله زیر را حل کنیم:

        dT/dt = alpha * d^2T/dx^2

شرایط مرزی: T(0,t) = 0  و  T(L,t) = 0
شرط اولیه:  T(x,0) = sin(pi * x / L)
        """)

        # گام ۱: روش تفکیک متغیرها
        self.ask_socratic_step(
            title='گام ۱: ایده اصلی حل',
            question='برای حل این معادله، فرض می‌کنیم پاسخ حاصل‌ضرب دو تابع جداگانه T(x,t) = X(x) * T_time(t) است.\nاسم این تکنیک چیست؟',
            target_expr=['تفکیک', 'separation'],
            hints=[
                'نام روش به جدا کردن متغیر مکانی (x) از زمان (t) اشاره دارد.',
                'کلمه اول آن "تفکیک" است...',
                'نام تکنیک: تفکیک متغیرها (Separation of Variables).',
            ],
            concept_key='separation_of_variables',
            validation_type='keyword',
        )

        # گام ۲: حل بخش مکانی
        target_X = sp.sympify(
            'sin(n * pi * x / L)', locals=self.symbols_dict
        )
        self.ask_socratic_step(
            title='گام ۲: بخش مکانی پاسخ X(x)',
            question='با اعمال شرایط مرزی T(0,t)=0 و T(L,t)=0، بخش مکانی چه تابع مثلثاتی بر حسب n, pi, x, L می‌شود؟\n(مثال برای قالب ورودی: sin(n*pi*x/L))',
            target_expr=target_X,
            concept_key='spatial_boundary',
            concept_keywords=['سینوس', 'sin', 'مثلثاتی', 'فاصله'],
            hints=[
                'فکر کنید کدام تابع مثلثاتی در نقطه x=0 برابر zero می‌شود؟',
                'شرط X(L)=0 باعث می‌شود آرگومان سینوس به صورت sin(n*pi*x/L) شکل بگیرد.',
                'پاسخ درست: sin(n*pi*x/L)',
            ],
        )

        # گام ۳: حل بخش زمانی
        target_time = sp.sympify(
            'exp(-alpha * (n * pi / L)**2 * t)', locals=self.symbols_dict
        )
        self.ask_socratic_step(
            title='گام ۳: بخش زمانی پاسخ T_time(t)',
            question='معادله بخش زمانی به صورت dT/dt + alpha*(n*pi/L)^2 * T = 0 است.\nپاسخ این معادله نمایی را بنویسید (از exp استفاده کنید):',
            target_expr=target_time,
            concept_key='exponential_decay',
            concept_keywords=['نمایی', 'توان', 'exp', 'کاهشی'],
            hints=[
                "جواب عمومی معادله y' + k*y = 0 به صورت exp(-k*t) است.",
                'در اینجا k = alpha * (n*pi/L)^2 می‌باشد.',
                'پاسخ درست: exp(-alpha * (n*pi/L)**2 * t)',
            ],
        )

        # گام ۴: ترکیب و پاسخ نهایی
        target_final = sp.sympify(
            'sin(pi * x / L) * exp(-alpha * (pi / L)**2 * t)',
            locals=self.symbols_dict,
        )
        self.ask_socratic_step(
            title='گام ۴: اعمال شرط اولیه و ساخت پاسخ نهایی',
            question='شرط اولیه ما T(x,0) = sin(pi*x/L) است. بنابراین فقط n=1 باقی می‌ماند.\nپاسخ نهایی T(x,t) را بنویسید:',
            target_expr=target_final,
            concept_key='initial_condition',
            concept_keywords=['ضرب سینوس', 'ضرب', 'سینوس در نمایی'],
            hints=[
                'کافی است در عبارت حاصل‌ضرب بخش مکانی و زمانی، n=1 قرار دهید.',
                'عبارت شامل ضرب سینوس در تابع نمایی است.',
                'پاسخ نهایی: sin(pi*x/L) * exp(-alpha * (pi/L)**2 * t)',
            ],
        )

        # اجرای شبیه‌سازی تعاملی با PINN
        self.run_pinn_verification_demo(target_final)


if __name__ == '__main__':
    tutor = SocraticMathTutor()
    tutor.start()