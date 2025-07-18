import sys
import json
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os

# 📌 경로: 모델 파일이 ���치한 폴더
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models")

# 4개 모델의 경로를 모두 정의합니다.
XGB_INCOME_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_income_change_model.pkl")
CAT_INCOME_MODEL_PATH = os.path.join(MODEL_DIR, "cat_income_change_model.cbm")
XGB_SATIS_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_satisfaction_change_model.pkl")
CAT_SATIS_MODEL_PATH = os.path.join(MODEL_DIR, "cat_satisfaction_change_model.cbm")

# 📌 soft-blending 비율 (XGBoost 모델의 비중)
ALPHA = 0.3

# 📌 4개 모델 모두 로드
try:
    xgb_income = joblib.load(XGB_INCOME_MODEL_PATH)
    cat_income = CatBoostRegressor()
    cat_income.load_model(CAT_INCOME_MODEL_PATH)

    xgb_satis = joblib.load(XGB_SATIS_MODEL_PATH)
    cat_satis = CatBoostRegressor()
    cat_satis.load_model(CAT_SATIS_MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {e}", file=sys.stderr)
    sys.exit(1)

# 📌 모델별 입력 feature 목록을 정확히 정의
# 소득 모델은 'job_satisfaction'을 포함
income_features = [
    "age", "gender", "education", "monthly_income", "job_category",
    "satis_wage", "satis_stability", "satis_task_content", "satis_work_env",
    "satis_work_time", "satis_growth", "satis_communication",
    "satis_fair_eval", "satis_welfare", "job_satisfaction",
    "prev_job_satisfaction", "prev_monthly_income", "job_category_income_avg",
    "income_relative_to_job", "job_category_education_avg", "education_relative_to_job",
    "job_category_satisfaction_avg", "age_x_job_category", "monthly_income_x_job_category",
    "education_x_job_category", "income_relative_to_job_x_job_category"
]

# 만족도 모델은 'job_satisfaction'을 포함하지 않음
satis_features = [
    "age", "gender", "education", "monthly_income", "job_category",
    "satis_wage", "satis_stability", "satis_task_content", "satis_work_env",
    "satis_work_time", "satis_growth", "satis_communication",
    "satis_fair_eval", "satis_welfare",
    "prev_job_satisfaction", "prev_monthly_income", "job_category_income_avg",
    "income_relative_to_job", "job_category_education_avg", "education_relative_to_job",
    "job_category_satisfaction_avg", "age_x_job_category", "monthly_income_x_job_category",
    "education_x_job_category", "income_relative_to_job_x_job_category"
]

# 📌 JSON 입력 파싱
def read_input():
    input_str = sys.stdin.read()
    return json.loads(input_str)

# 📌 예측 함수
def predict_scenario(row):
    # 모델별로 필요한 변수만 선택하여 별도의 DataFrame 생성
    income_df = pd.DataFrame([row])[income_features]
    satis_df = pd.DataFrame([row])[satis_features]

    # CatBoost 모델 예측
    income_cat_pred = cat_income.predict(income_df)[0]
    satis_cat_pred = cat_satis.predict(satis_df)[0]

    # XGBoost 모델 예측
    income_xgb_pred = xgb_income.predict(income_df)[0]
    satis_xgb_pred = xgb_satis.predict(satis_df)[0]

    # Soft Blending
    income_pred = ALPHA * income_xgb_pred + (1 - ALPHA) * income_cat_pred
    satis_pred = ALPHA * satis_xgb_pred + (1 - ALPHA) * satis_cat_pred

    # 만족도 예측값 후처리: 값 범위만 제한하고 정수로의 반올림은 제거
    MIN_SATISFACTION_CHANGE = -4
    MAX_SATISFACTION_CHANGE = 3
    # np.round()를 제거하여 미세한 예측값(e.g., 0.25)이 0으로 변하는 것을 방지
    satis_pred_processed = np.clip(satis_pred, MIN_SATISFACTION_CHANGE, MAX_SATISFACTION_CHANGE)

    # 소득과 동일하게 소수점 4자리까지 반올림하여 반환
    return round(income_pred, 4), round(satis_pred_processed, 4)

# 📌 메인 실행 로직
if __name__ == "__main__":
    try:
        scenarios_data = read_input()
        results = []
        for scenario in scenarios_data:
            income_change, satis_change = predict_scenario(scenario)
            results.append({
                "income_change_rate": income_change,
                "satisfaction_change_score": satis_change
            })
        print(json.dumps(results))
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)
