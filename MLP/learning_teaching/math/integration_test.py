import torch
import numpy as np
# غیرفعال کردن لایه گرافیکی GUI جهت جلوگیری از خطای Tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pinn import train_pinn

# ۱. تعریف تابع پاسخ دقیق بر اساس خروجی SymPy
def exact_solution(x, t, alpha=0.01, L=1.0):
    # T(x,t) = sin(pi * x / L) * exp(-alpha * (pi / L)^2 * t)
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L)**2 * t)

# ۲. تست شبکه عصبی آموزش‌دیده
def test_and_evaluate(model, alpha=0.01, L=1.0):
    model.eval()
    
    # تعریف ۱۰۰ نقطه تست در راستای مکان برای زمان t = 0.1
    x_test = np.linspace(0, L, 100).reshape(-1, 1)
    t_test = np.full_like(x_test, 0.1)  # t = 0.1
    
    # تبدیل به تانسور PyTorch
    x_tensor = torch.tensor(x_test, dtype=torch.float32)
    t_tensor = torch.tensor(t_test, dtype=torch.float32)
    
    # پیش‌بینی PINN
    with torch.no_grad():
        T_pinn = model(x_tensor, t_tensor).numpy()
    
    # پاسخ دقیق SymPy
    T_exact = exact_solution(x_test, 0.1, alpha, L)
    
    # محاسبه میانگین قدر مطلق خطا (MAE)
    mae_error = np.mean(np.abs(T_pinn - T_exact))
    print(f"\n==========================================")
    print(f"Mean Absolute Error (MAE) at t=0.1: {mae_error:.6f}")
    print(f"==========================================")
    
    # ۳. رسم نمودار مقایسه
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, T_exact, 'r-', label='Exact (SymPy)', linewidth=2)
    plt.plot(x_test, T_pinn, 'b--', label='PINN Prediction', linewidth=2)
    plt.title(f'Comparison at t = 0.1s (MAE: {mae_error:.6f})')
    plt.xlabel('x (Position)')
    plt.ylabel('T (Temperature)')
    plt.legend()
    plt.grid(True)
    plt.savefig("pinn_result.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("نمودار خروجی در فایل pinn_result.png ذخیره شد.")

# اجرا: ابتدا مدل آموزش می‌بیند و سپس تست می‌شود
if __name__ == "__main__":
    trained_model = train_pinn()
    test_and_evaluate(trained_model)