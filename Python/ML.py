import pandas as pd
import os
import ConnectDB

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor


projectRoot = "D:/GAIP"
trainInputDir = "resources/dev02/trainData/inputData"
trainTargetDir = "resources/dev02/trainData/targetData"
testInputDir = "resources/dev02/testData/inputData"
testTargetDir = "resources/dev02/testData/targetData"
encoding = "utf-8"
columnList = ["workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count","minSalary","maxSalary","meanSalary"]


def loadAllData(directory, columnList):
    fileNames = os.listdir(directory)

    df = pd.DataFrame(columns=columnList)
    for fName in fileNames:
        path = f"{directory}/{fName}"
        newDf = pd.read_csv(path, encoding=encoding)

        df = pd.concat([df, newDf])
    return df

def testML():
    inputDataDir = f"{projectRoot}/{trainInputDir}"
    targetDataDir = f"{projectRoot}/{trainTargetDir}"

    trainInput = loadAllData(inputDataDir, columnList).values
    trainTarget = loadAllData(targetDataDir, columnList).values
    
    inputDataDir = f"{projectRoot}/{testInputDir}"
    targetDataDir = f"{projectRoot}/{testTargetDir}"

    testInput = loadAllData(inputDataDir, columnList).values
    testTarget = loadAllData(targetDataDir, columnList).values

    LinearML(trainInput, trainTarget, testInput, testTarget)
    RandomForestML(trainInput, trainTarget, testInput, testTarget)
    # RandomForestML(trainInput, trainTarget, testInput, testTarget)
    RidgeML(trainInput, trainTarget, testInput, testTarget)
    LassoML(trainInput, trainTarget, testInput, testTarget)
    


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
    

def RandomForestML(trainInput, trainTarget, testInput, testTarget):
    rfr = RandomForestRegressor(
            # n_estimators=100,               # 생성할 트리 개수
            # criterion='squared_error',      # 분할 기준 (손실 함수: 평균제곱오차 MSE)
            # max_depth=None,                 # 트리 최대 깊이 제한 (None이면 끝까지 분할)
            # min_samples_split=2,            # 내부 노드 분할에 필요한 최소 샘플 수
            # min_samples_leaf=1,             # 리프 노드에 있어야 할 최소 샘플 수
            # min_weight_fraction_leaf=0.0,   # 리프 노드에 있어야 할 최소 가중치 비율 (샘플 가중치 쓸 때 사용)
            # max_features=1.0,               # 분할에 사용할 피처 비율 (1.0이면 전체 피처 사용)
            # max_leaf_nodes=None,            # 리프 노드 최대 개수 제한 (None이면 제한 없음)
            # min_impurity_decrease=0.0,      # 이 값보다 손실 감소가 작으면 분할하지 않음 (분할 최소 기준)
            # bootstrap=True,                 # 부트스트랩 샘플링 사용 여부 (True면 중복 허용)
            # oob_score=False,                # Out-of-Bag 샘플로 일반화 성능 평가할지 여부
            # n_jobs=None,                    # 사용할 CPU 코어 수 (None이면 1, -1이면 전체)
            # random_state=42,                # 랜덤 시드 (재현 가능성 위해 고정)
            # verbose=0,                      # 학습 과정 출력 수준 (0이면 출력 없음)
            # warm_start=False,               # 기존 트리에 이어서 추가 학습할지 여부
            # ccp_alpha=0.0,                  # Minimal cost-complexity pruning 강도 (0이면 가지치기 안함)
            # max_samples=None                # 부트스트랩 샘플 수 제한 (None이면 전체 샘플 사용)
        )
    rfr.fit(trainInput, trainTarget)

    # train
    print(rfr.score(trainInput, trainTarget))

    # test
    print(rfr.score(testInput, testTarget))

    # df = pd.DataFrame(rfr.predict(testInput), columns=columnList)
    # df = df.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"float64"})
    # print(df)

testML()
