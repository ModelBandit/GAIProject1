import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 1. 데이터 불러오기
df = pd.read_csv("klips_data_21.csv")

# 2. Feature / Target 설정
features = [
    "age", "gender", "education", "monthly_income", "job_category",
    "satis_wage", "satis_stability", "satis_task_content", "satis_work_env",
    "satis_work_time", "satis_growth", "satis_communication",
    "satis_fair_eval", "satis_welfare", "prev_job_satisfaction",
    "prev_monthly_income", "job_category_income_avg",
    "income_relative_to_job", "job_category_education_avg", "education_relative_to_job",
    "job_category_satisfaction_avg",
    "age_x_job_category", "monthly_income_x_job_category",
    "education_x_job_category", "income_relative_to_job_x_job_category"
]

X = df[features]
y = df["satisfaction_change_score"]

# 사용자께서 확인해주신 실제 y값의 최소/최대 범위 사용
MIN_SATISFACTION_CHANGE = -4
MAX_SATISFACTION_CHANGE = 3
print(f"Satisfaction Change Score의 실제 최소값: {MIN_SATISFACTION_CHANGE}")
print(f"Satisfaction Change Score의 실제 최대값: {MAX_SATISFACTION_CHANGE}")


# 3. 훈련/테스트 분리
latest_year = df["year"].max()
X_train = X[df["year"] < latest_year]
y_train = y[df["year"] < latest_year]
X_test = X[df["year"] == latest_year]
y_test = y[df["year"] == latest_year]

# 4. XGBoost 하이퍼파라미터 튜닝
tscv = TimeSeriesSplit(n_splits=5)
xgb_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=xgb_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)

# 5. CatBoost 하이퍼파라미터 튜닝
cat_grid = {
    'iterations': [50, 100, 150],
    'depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2]
}
cat_search = GridSearchCV(
    estimator=CatBoostRegressor(verbose=0, random_state=42),
    param_grid=cat_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
cat_search.fit(X_train, y_train, cat_features=["gender", "education", "job_category"])
cat_best = cat_search.best_estimator_
y_pred_cat = cat_best.predict(X_test)

# 6. Soft-Blending
alpha = 0.3
y_pred_blend = alpha * y_pred_xgb + (1 - alpha) * y_pred_cat

# --- 예측값 후처리: 소수점을 정수로 변환 및 실제 데이터 범위로 클리핑 ---
# 1. 반올림 (가장 가까운 정수)
y_pred_xgb_int = np.round(y_pred_xgb)
y_pred_cat_int = np.round(y_pred_cat)
y_pred_blend_int = np.round(y_pred_blend)

# 2. 값 범위 제한 (클리핑): 사용자께서 확인해주신 최소값과 최대값으로 조정
y_pred_xgb_int = np.clip(y_pred_xgb_int, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)
y_pred_cat_int = np.clip(y_pred_cat_int, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)
y_pred_blend_int = np.clip(y_pred_blend_int, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)
# ----------------------------------------------------

# 7. 평가 함수 (이전과 동일)
def evaluate(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n-------- {name} --------")
    print("📉 RMSE:", round(rmse, 4))
    print("📉 MAE :", round(mae, 4))
    print("📈 R²  :", round(r2, 4))

# 8. 결과 출력 (후처리된 예측값으로 평가)
evaluate("XGBoost (Tuned, Integer)", y_test, y_pred_xgb_int)
evaluate("CatBoost (Tuned, Integer)", y_test, y_pred_cat_int)
evaluate("Soft-Blended Ensemble (Integer)", y_test, y_pred_blend_int)

import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost Feature Importance
def plot_xgb_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi, palette='Greens_d')
    plt.title('XGBoost Feature Importance (Top {})'.format(top_n))
    plt.tight_layout()
    plt.show()

# CatBoost Feature Importance
def plot_cat_importance(model, feature_names, top_n=15):
    importances = model.get_feature_importance()
    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi, palette='Blues_d')
    plt.title('CatBoost Feature Importance (Top {})'.format(top_n))
    plt.tight_layout()
    plt.show()

# 호출 예시 (이미 학습된 모델과 feature list 사용)
plot_xgb_importance(xgb_best, X.columns)
plot_cat_importance(cat_best, X.columns)

import joblib
import os

# 저장 디렉토리 생성 (필요 시)
os.makedirs("saved_models", exist_ok=True)

# 1. XGBoost 모델 저장 (.pkl)
xgb_path = "saved_models/xgb_satisfaction_change_model.pkl"
joblib.dump(xgb_best, xgb_path)
print(f"[저장 완료] XGBoost 모델 → {xgb_path}")

# 2. CatBoost 모델 저장 (.cbm)
cat_path = "saved_models/cat_satisfaction_change_model.cbm"
cat_best.save_model(cat_path)
print(f"[저장 완료] CatBoost 모델 → {cat_path}")

# 3. Soft-Blending 모델은 따로 저장하지 않음
# -> 자바에서 XGBoost/CatBoost 두 모델을 불러와 예측값 blending 처리