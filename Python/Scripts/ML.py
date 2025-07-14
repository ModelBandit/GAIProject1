import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



projectRoot = "./"
trainInputDir = "resources/dev02/trainData/inputData"
trainTargetDir = "resources/dev02/trainData/targetData"
testInputDir = "resources/dev02/testData/inputData"
testTargetDir = "resources/dev02/testData/targetData"
encoding = "utf-8"
# columnList = ["maleCount","femaleCount","ageLt40Count","ageGte40Count"]
columnList = ["workerCount", "maleCount","femaleCount","ageLt40Count","ageGte40Count", "minSalary", "maxSalary", "meanSalary"]
targetColumnList = ["companyCount", "workerCount"] # "companyCount",


def loadAllData(directory, columnList):
    fileNames = os.listdir(directory)

    df = pd.DataFrame(columns=columnList)
    for fName in fileNames:
        path = f"{directory}/{fName}"
        newDf = pd.read_csv(path, encoding=encoding)[columnList]

        df = pd.concat([df, newDf])
    return df

def testML():
    # 나눠진 경우
    # inputDataDir = f"{projectRoot}/{trainInputDir}"
    # targetDataDir = f"{projectRoot}/{trainTargetDir}"

    # trainInput = loadAllData(inputDataDir, columnList)#.values
    # trainTarget = loadAllData(targetDataDir, targetColumnList)#.values
    
    # inputDataDir = f"{projectRoot}/{testInputDir}"
    # targetDataDir = f"{projectRoot}/{testTargetDir}"

    # testInput = loadAllData(inputDataDir, columnList)#.values
    # testTarget = loadAllData(targetDataDir, targetColumnList)#.values

    inputDataDir = r".\resources\dev02\haveSalary\inputData"
    targetDataDir = r".\resources\dev02\haveSalary\targetData"
    inputDf = loadAllData(inputDataDir, columnList)
    targetDf = loadAllData(targetDataDir, targetColumnList)

    # 섞으려고 추가함
    # inputDf = pd.concat([trainInput, testInput])
    # targetDf = pd.concat([trainTarget, testTarget])


    # corr_matrix = pd.DataFrame(inputDf, columns=inputDf.columns).corr()
    # plt.figure(figsize=(200,200))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.title("Feature Correlation Matrix")
    # plt.show()


    trainInput, testInput, trainTarget, testTarget = train_test_split(inputDf, targetDf, test_size=0.2, random_state=42)
    trainInput = trainInput.astype({columnList[0]:int, columnList[1]:int,columnList[2]:int,columnList[3]:int,columnList[4]:int, columnList[5]:int, columnList[6]:int, columnList[7]:int})
    testInput = testInput.astype({columnList[0]:int, columnList[1]:int,columnList[2]:int,columnList[3]:int,columnList[4]:int  , columnList[5]:int, columnList[6]:int, columnList[7]:int})
    trainTarget = trainTarget.astype({targetColumnList[0]:int, targetColumnList[1]:int})
    testTarget = testTarget.astype({targetColumnList[0]:int, targetColumnList[1]:int})

    df_corr = trainInput.copy()
    df_corr['target'] = trainTarget[targetColumnList[1]]

    target_corr = df_corr.corr()['target'].drop('target').sort_values(ascending=False)
    print(target_corr)

    # trainTarget = trainTarget[targetColumnList[0]].values.ravel()
    # testTarget = testTarget[targetColumnList[0]].values.ravel()



    # lML = LinearML(trainInput, trainTarget, testInput, testTarget)

    
    # PolynomialLinearML(trainInput, trainTarget, testInput, testTarget)
    # RidgeML(trainInput, trainTarget, testInput, testTarget)ExtraRandomForestML
    # LassoML(trainInput, trainTarget, testInput, testTarget)
    
    # dtML = DecisionTreeML(trainInput, trainTarget, testInput, testTarget)
    # rfML = RandomForestML(trainInput, trainTarget, testInput, testTarget)
    # erfML = ExtraRandomForestML(trainInput, trainTarget, testInput, testTarget)
    # gbr = GradientBoostingRegressorML(trainInput, trainTarget, testInput, testTarget)
    # hgbr = HistGradientBoostingRegressorML(trainInput, trainTarget, testInput, testTarget)
    # # permutation_importance_ML(trainInput, trainTarget, testInput, testTarget)

    # xg = XGBRegressor_ML(trainInput, trainTarget, testInput, testTarget)
    # # XGBRFRegressor_ML(trainInput, trainTarget, testInput, testTarget)

    # lg = LGBMRegressor_ML(trainInput, trainTarget, testInput, testTarget)

    
    # categories =["DecisionTreeML","RandomForestML","ExtraRandomForestML","GradientBoostingRegressorML","HistGradientBoostingRegressorML","XGBRegressor_ML","LGBMRegressor_ML"]

    # values1 = [dtML[0], rfML[0], erfML[0], gbr[0], hgbr[0], xg[0], lg[0]]
    # values2 = [dtML[1], rfML[1], erfML[1], gbr[1], hgbr[1], xg[1], lg[1]]

    # x = np.arange(len(categories))  # x축 위치 [0, 1, 2, ..., 19]
    # width = 0.05  # 바 폭
    
    # plt.bar(x - width, values1, 0.1, label='Train')
    # plt.bar(x + width, values2, 0.1, label='Test')
    # plt.xticks(x, categories, rotation=45)  # x축 이름 및 회전
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    


def LinearML(trainInput, trainTarget, testInput, testTarget):

    # print(trainInput.shape)
    # print(trainTarget.shape)
    print("LinearML")
    lr = LinearRegression(
        # * 인자전달은 반드시 키워드 인자로 전달할 것 (그런데 뭔가 다른가봄)
        # fit_intercept=True,     # True일 경우 절편(bias, intercept)을 학습함 (y = wx + b). False면 b 생략됨.
        # copy_X=True,            # X를 복사할지 여부. True면 원본 X는 변경되지 않음.
        # tol=1e-6,               # 반복 수치 해석 알고리즘의 수렴 허용 오차 (작을수록 정확하지만 느림)
        # n_jobs=None,            # 병렬 연산에 사용할 CPU 코어 수 (None이면 1, -1이면 전체 사용)
        # positive=False          # True로 설정하면 회귀 계수를 음수가 아닌 값으로 강제함 (비음수 회귀)
    )
    lr.fit(trainInput, trainTarget)

    scoreList = [lr.score(trainInput, trainTarget), lr.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList
    

    # df = pd.DataFrame(lr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
def DecisionTreeML(trainInput, trainTarget, testInput, testTarget):

    # print(trainInput.shape)
    # print(trainTarget.shape)
    print("DecisionTreeML")
    lr = DecisionTreeRegressor()
    lr.fit(trainInput, trainTarget)
    scoreList = [lr.score(trainInput, trainTarget), lr.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])


    # plt.figure(figsize=(10,7))
    # plot_tree(lr, max_depth=1, filled=True)
    # plt.show()
    return scoreList
    

    # df = pd.DataFrame(lr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)

def PolynomialLinearML(trainInput, trainTarget, testInput, testTarget):

    # print(trainInput.shape)
    # print(trainTarget.shape)
    print("PolynomialLinearML")
    lr = make_pipeline(PolynomialFeatures(degree=4, include_bias=False), LinearRegression())
    lr.fit(trainInput, trainTarget)

    # train
    print(lr.score(trainInput, trainTarget))

    # test
    print(lr.score(testInput, testTarget))
    

    # df = pd.DataFrame(lr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)

def RidgeML(trainInput, trainTarget, testInput, testTarget):

    print("RidgeML")
    
    alphaList = [0.001,0.01,0.1,1,10,100]
    for alpha in alphaList:
        print(f"RidgeML - alpha: {alpha}")
        ridge = Ridge(alpha=alpha)
        ridge.fit(trainInput, trainTarget)

        # train
        print(ridge.score(trainInput, trainTarget))

        # test
        print(ridge.score(testInput, testTarget))

def LassoML(trainInput, trainTarget, testInput, testTarget):

    print("LassoML")
    
    alphaList = [0.001,0.01,0.1,1,10,100]
    for alpha in alphaList:
        print(f"LassoML - alpha: {alpha}")
        lasso = Lasso(alpha=alpha)
        lasso.fit(trainInput, trainTarget)

        # train
        print(lasso.score(trainInput, trainTarget))

        # test
        print(lasso.score(testInput, testTarget))
    
est = 300
mDepth = 15
msl = 1
def RandomForestML(trainInput, trainTarget, testInput, testTarget):

    print("RandomForestML")

    rfr = RandomForestRegressor(
            n_estimators=est,               # 생성할 트리 개수
            # criterion='squared_error',      # 분할 기준 (손실 함수: 평균제곱오차 MSE)
            max_depth=mDepth,                 # 트리 최대 깊이 제한 (None이면 끝까지 분할)
            # min_samples_split=2,            # 내부 노드 분할에 필요한 최소 샘플 수
            min_samples_leaf=msl,             # 리프 노드에 있어야 할 최소 샘플 수
            # min_weight_fraction_leaf=0.0,   # 리프 노드에 있어야 할 최소 가중치 비율 (샘플 가중치 쓸 때 사용)
            max_features=0.8,               # 분할에 사용할 피처 비율 (1.0이면 전체 피처 사용)
            # max_leaf_nodes=None,            # 리프 노드 최대 개수 제한 (None이면 제한 없음)
            # min_impurity_decrease=0.0,      # 이 값보다 손실 감소가 작으면 분할하지 않음 (분할 최소 기준)
            bootstrap=True,                 # 부트스트랩 샘플링 사용 여부 (True면 중복 허용)
            # oob_score=False,                # Out-of-Bag 샘플로 일반화 성능 평가할지 여부
            n_jobs=-1,                    # 사용할 CPU 코어 수 (None이면 1, -1이면 전체)
            random_state=42,                # 랜덤 시드 (재현 가능성 위해 고정)
            # verbose=0,                      # 학습 과정 출력 수준 (0이면 출력 없음)
            # warm_start=False,               # 기존 트리에 이어서 추가 학습할지 여부
            # ccp_alpha=0.0,                  # Minimal cost-complexity pruning 강도 (0이면 가지치기 안함)
            # max_samples=None,                # 부트스트랩 샘플 수 제한 (None이면 전체 샘플 사용)
        )
    # params = {
    #     'n_estimators': range(50,300,5),
    #     'max_depth': range(5,20),
    #     'min_samples_split': range(2,10),
    #     'min_samples_leaf': [1, 2],
    #     'max_features': np.arange(0,1.0,0.05)
    # }
    # gs = GridSearchCV(rfr, params, n_jobs=-1)
    # gs.fit(trainInput, trainTarget)
    # print(gs.best_params_)
    # print(gs.best_score_)

    rfr.fit(trainInput, trainTarget)
    
    # resultInput = permutation_importance(rfr, trainInput, trainTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultInput.importances_mean)

    # resultTarget = permutation_importance(rfr, testInput, testTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultTarget.importances_mean)
    # train


    importances = rfr.feature_importances_

    # for name, score in zip(trainInput.columns, importances):
    #     print(f"{name}: {score:.4f}")

    scoreList = [rfr.score(trainInput, trainTarget), rfr.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList

    # df = pd.DataFrame(rfr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)

from sklearn.ensemble import ExtraTreesRegressor
def ExtraRandomForestML(trainInput, trainTarget, testInput, testTarget):

    print("ExtraRandomForestML")

    rfr = ExtraTreesRegressor(
                                n_estimators=est, 
                                # criterion='squared_error',
                                max_depth=mDepth, 
                                # min_samples_split=2, 
                                min_samples_leaf=msl, 
                                # min_weight_fraction_leaf=0.0, 
                                max_features=0.8, 
                                # max_leaf_nodes=None, 
                                # min_impurity_decrease=0.0, 
                                bootstrap=True,
                                # oob_score=False, 
                                n_jobs=-1, random_state=42, 
                                # verbose=0, 
                                # warm_start=False, 
                                # ccp_alpha=0.0, 
                                # max_samples=None, 
                                # monotonic_cst=None,
            )
    rfr.fit(trainInput, trainTarget)

    # resultInput = permutation_importance(rfr, trainInput, trainTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultInput.importances_mean)

    # resultTarget = permutation_importance(rfr, testInput, testTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultTarget.importances_mean)
    
    scoreList = [rfr.score(trainInput, trainTarget), rfr.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList


    # df = pd.DataFrame(rfr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)

from sklearn.ensemble import GradientBoostingRegressor
def GradientBoostingRegressorML(trainInput, trainTarget, testInput, testTarget):

    print("GradientBoostingRegressorML")

    gbr = GradientBoostingRegressor(
        loss='squared_error',           # 손실 함수: 일반 회귀에서는 평균제곱오차(MSE)
        learning_rate=0.1,              # 각 트리의 기여도 크기 (작을수록 더 천천히 학습, 과적합 억제)
        n_estimators=100,               # 부스팅할 트리의 수 (많을수록 모델 복잡도 증가)
        subsample=1.0,                  # 각 트리 학습 시 사용할 샘플 비율 (1.0이면 전체 사용, 0.5~0.9는 과적합 방지)
        criterion='friedman_mse',       # 트리 분할 품질 기준 (기본값은 `friedman_mse`로 MSE보다 일반화 성능 좋음)
        min_samples_split=2,            # 내부 노드를 분할하기 위한 최소 샘플 수
        min_samples_leaf=1,             # 리프 노드에 있어야 할 최소 샘플 수 (과적합 방지하려면 3~5 추천)
        min_weight_fraction_leaf=0.0,   # 리프 노드에 있어야 할 샘플 가중치 최소 비율 (클수록 가지치기 강함)
        max_depth=3,                    # 각 결정 트리의 최대 깊이 (과적합 방지 핵심 파라미터)
        min_impurity_decrease=0.0,      # 분할이 유효하려면 줄어들어야 하는 최소 불순도 (값을 높이면 가지치기)
        init=None,                      # 초기 모델 지정 (보통은 None으로 자동 설정)
        random_state=None,              # 랜덤 시드 (재현성을 위해 고정 가능)
        max_features=None,              # 분할 시 고려할 피처 수 ('sqrt', 0.8 등으로 설정 시 과적합 억제)
        alpha=0.9,                      # Quantile loss일 때만 사용 (기본 MSE에서는 영향 없음)
        verbose=0,                      # 학습 중 출력 수준 (0=조용, 1=간단 출력)
        max_leaf_nodes=None,           # 트리의 최대 리프 노드 수 (None이면 제한 없음)
        warm_start=False,              # 이전 학습 결과를 이어서 학습할지 여부 (True로 설정 시 점진적 학습 가능)
        validation_fraction=0.1,       # 조기 종료 검증용 데이터 비율 (early stopping용)
        n_iter_no_change=None,         # early stopping을 위한 반복 횟수 기준 (None이면 사용 안 함)
        tol=0.0001,                     # early stopping 기준 수렴 허용 오차
        ccp_alpha=0.0                  # Minimal Cost-Complexity Pruning 강도 (가지치기)
    )
    gbr.fit(trainInput, trainTarget)
    
    # resultInput = permutation_importance(gbr, trainInput, trainTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultInput.importances_mean)

    # resultTarget = permutation_importance(gbr, testInput, testTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultTarget.importances_mean)

    scoreList = [gbr.score(trainInput, trainTarget), gbr.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList

    # df = pd.DataFrame(gbr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)

from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
def HistGradientBoostingRegressorML(trainInput, trainTarget, testInput, testTarget):

    print("HistGradientBoostingRegressorML")

    gbr = HistGradientBoostingRegressor(
        loss='squared_error', 
        quantile=None, learning_rate=0.1,
        max_iter=100, 
        max_leaf_nodes=31, 
        max_depth=None, 
        min_samples_leaf=20, 
        l2_regularization=0.0, 
        max_features=1.0, 
        max_bins=255, 
        categorical_features='from_dtype', 
        monotonic_cst=None, 
        interaction_cst=None, 
        warm_start=False, 
        early_stopping='auto', 
        scoring='loss', 
        validation_fraction=0.1, 
        n_iter_no_change=10, 
        tol=1e-07, 
        verbose=0, 
        random_state=None
    )
    gbr.fit(trainInput, trainTarget)
    

    scoreList = [gbr.score(trainInput, trainTarget), gbr.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList

    # df = pd.DataFrame(gbr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)

from sklearn.inspection import permutation_importance
def permutation_importance_ML(trainInput, trainTarget, testInput, testTarget):

    print("permutation_importance_ML")

    hgbr = HistGradientBoostingRegressor()
    hgbr.fit(trainInput, trainTarget)
    
    resultInput = permutation_importance(hgbr, trainInput, trainTarget, 
                                    n_repeats=10, n_jobs=-1)
    print(resultInput.importances_mean)

    resultTarget = permutation_importance(hgbr, testInput, testTarget, 
                                    n_repeats=10, n_jobs=-1)
    print(resultTarget.importances_mean)
    

    scoreList = [hgbr.score(trainInput, trainTarget), hgbr.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList

from xgboost import XGBRegressor
def XGBRegressor_ML(trainInput, trainTarget, testInput, testTarget):

    print("XGBRegressor_ML")

    xgb = XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=200, early_stopping_rounds=10, max_feature=0.8)
    xgb.fit(trainInput, trainTarget, eval_set=[(testInput, testTarget)])
    
    resultInput = permutation_importance(xgb, trainInput, trainTarget, 
                                    n_repeats=10, n_jobs=-1)
    print(resultInput.importances_mean)

    resultTarget = permutation_importance(xgb, testInput, testTarget, 
                                    n_repeats=10, n_jobs=-1)
    print(resultTarget.importances_mean)
    

    scoreList = [xgb.score(trainInput, trainTarget), xgb.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList

from xgboost import XGBRFRegressor
def XGBRFRegressor_ML(trainInput, trainTarget, testInput, testTarget):

    print("XGBRFRegressor_ML")

    xgb = XGBRFRegressor(learning_rate=0.05, max_depth=10, n_estimators=200, max_feature=0.8)
    xgb.fit(trainInput, trainTarget, eval_set=[(testInput, testTarget)])
    
    # resultInput = permutation_importance(xgb, trainInput, trainTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultInput.importances_mean)

    # resultTarget = permutation_importance(xgb, testInput, testTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultTarget.importances_mean)
    

    scoreList = [xgb.score(trainInput, trainTarget), xgb.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList

from lightgbm import LGBMRegressor
def LGBMRegressor_ML(trainInput, trainTarget, testInput, testTarget):

    print("LGBMRegressor_ML")

    xgb = LGBMRegressor(
        max_depth=10,             # 트리 깊이 제한
        num_leaves=5,           # 리프 수
        learning_rate=0.05,
        n_estimators=200,
        max_feature=0.8
    )
    xgb.fit(trainInput, trainTarget)
    
    # resultInput = permutation_importance(xgb, trainInput, trainTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultInput.importances_mean)

    # resultTarget = permutation_importance(xgb, testInput, testTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultTarget.importances_mean)
    
    scoreList = [xgb.score(trainInput, trainTarget), xgb.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList


from lightgbm import DaskLGBMRegressor
def DaskLGBMRegressor_ML(trainInput, trainTarget, testInput, testTarget):

    print("DaskLGBMRegressor_ML")

    xgb = DaskLGBMRegressor(
        max_depth=10,             # 트리 깊이 제한
        num_leaves=5,           # 리프 수
        learning_rate=0.05,
        n_estimators=200,
        max_feature=0.8
    )
    xgb.fit(trainInput, trainTarget)
    
    # resultInput = permutation_importance(xgb, trainInput, trainTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultInput.importances_mean)

    # resultTarget = permutation_importance(xgb, testInput, testTarget, 
    #                                 n_repeats=10, n_jobs=-1)
    # print(resultTarget.importances_mean)
    
    scoreList = [xgb.score(trainInput, trainTarget), xgb.score(testInput, testTarget)]
    
    print(scoreList[0])
    print(scoreList[1])

    return scoreList

if __name__ == "__main__":
    testML()
