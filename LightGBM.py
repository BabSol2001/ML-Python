import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ۱. ساخت یک داده نمونه بزرگ برای دیدن تفاوت سرعت
X, y = make_regression(n_samples=50000, n_features=30, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Standard GBR": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
}

print(f"{'Model':<15} | {'Time (sec)':<12} | {'Test MSE':<10}")
print("-" * 42)

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    print(f"{name:<15} | {elapsed_time:<12.4f} | {mse:<10.2f}")