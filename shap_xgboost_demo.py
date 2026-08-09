import matplotlib
matplotlib.use('Agg')  # جلوگیری از خطای گرافیکی در VS Code

import shap
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ۱. ساخت داده‌های نمونه با تنظیم صریح پارامترهای ویژگی
X_arr, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_informative=4, 
    n_redundant=0,      # اضافه شد تا مجموع ویژگی‌ها معتبر باشد
    n_repeated=0,       # اضافه شد
    random_state=42
)

feature_names = ['Age', 'Income', 'Credit_Score', 'Debt_Ratio', 'Years_Employed']
X = pd.DataFrame(X_arr, columns=feature_names)

# ۲. تقسیم داده‌ها و آموزش مدل XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# ۳. محاسبه مقادیر SHAP با الگوریتم بهینه‌شده TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# -------------------------------------------------------------
# ۴. رسم نمودار Summary Plot (تفسیر کلی مدل)
# -------------------------------------------------------------
plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot (Feature Impact)', fontsize=12, fontweight='bold', pad=15)
output_file1 = 'shap_summary_plot.png'
plt.savefig(output_file1, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ نمودار کلی SHAP در '{output_file1}' ذخیره شد.")

# -------------------------------------------------------------
# ۵. رسم نمودار Waterfall Plot (تفسیر پیش‌بینی برای اولین داده تست)
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_values[0], show=False)
plt.title('SHAP Waterfall Plot for Sample #0', fontsize=12, fontweight='bold', pad=15)
output_file2 = 'shap_waterfall_plot.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ نمودار تکی SHAP در '{output_file2}' ذخیره شد.")