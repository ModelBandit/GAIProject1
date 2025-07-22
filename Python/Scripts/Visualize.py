import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

from ML import loadAllData, DecisionTreeML
from sklearn.model_selection import train_test_split

projectRoot = "./"
resourcesDir = "resources/dev02/haveSalary/예외"
encoding = "utf-8"


engColumnList = ["industryType", 
              "companyCount", "ownerMaleRate","ownerFemaleRate", "singlePropCompanyRate", "multiBusinessCompanyRate", 
              "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
              "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",
              "workerCount", "workerMaleRate", "workerFemaleRate", "singlePropWorkerRate", "multiBusinessWorkerRate", 
              "selfEmpFamilyWorkerRate", "fulltimeWorkerRate", "dayWorkerRate", "etcWorkerRate",
              "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
              "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",
              "avgAge","avgServYear","avgWorkDay","avgTotalWorkTime","avgRegularWorkDay","avgOverWorkDay","avgSalary","avgFixedSalary","avgOvertimeSalary","avgBonusSalary"] 

# columnList = ["maleCount","femaleCount","ageLt40Count","ageGte40Count"]
columnList = [
              "companyCount", 
              "ownerMaleRate",
              "singlePropCompanyRate", 
            "multiBusinessCompanyRate",
              "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
              "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",
              "workerCount", 
              "workerMaleRate", "singlePropWorkerRate", 
              "selfEmpFamilyWorkerRate", "fulltimeWorkerRate", "dayWorkerRate", "etcWorkerRate",
              "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
              "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",
              "avgAge",
            "avgServYear",
            "avgWorkDay",
              "avgTotalWorkTime",
              "avgRegularWorkDay","avgOverWorkDay","avgSalary","avgFixedSalary","avgOvertimeSalary","avgBonusSalary"
            ]
# UD Comp 8개
# UD Worker 12~3개
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def loadAllData(directory, columnList):
    fileNames = os.listdir(directory)

    df = pd.DataFrame(columns=columnList)
    for fName in fileNames:
        path = f"{directory}/{fName}"
        newDf = pd.read_csv(path, encoding=encoding)[columnList]

        df = pd.concat([df, newDf])
    return df

def drawScatter():
    inputDataDir = r"resources\dev02\data"
    df = loadAllData(inputDataDir, columnList)
    
    sns.scatterplot(data=df, x="avgServYear", y="avgWorkDay")
    plt.title("avgServYear vs avgWorkDay")
    plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def drawPCA():
    inputDataDir = r"resources\dev02\data"
    # 예: 비율 변수들만 선택
    cols = ["multiBusinessCompanyRate", "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
              "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",]
    df = loadAllData(inputDataDir, cols)
    
    # 정규화 → PCA는 스케일에 민감함
    scaled = StandardScaler().fit_transform(df)

    # PCA 적용 (설명력 높은 주성분만 남김)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)

    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title("PCA 결과 시각화")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


from sklearn.cluster import KMeans
def drawKMeansElbow():
    inputDataDir = r"resources\dev02\data"
    # 예: 비율 변수들만 선택
    cols = ["singlePropCompanyRate", "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
              "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",]
    # cols = ["multiBusinessCompanyRate", "U1D5WorkerRate", "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
    #           "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",]
    df = loadAllData(inputDataDir, cols)

    # 정규화 → PCA는 스케일에 민감함
    scaled = StandardScaler().fit_transform(df)

    # PCA 적용 (설명력 높은 주성분만 남김)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    inertias = []
    for k in range(1, 21):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(pca_result)

    plt.plot(range(1, 21), inertias, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

customKeyCodeList =[ 
["농업임업및어업",
"광업"],
"제조업",
["도매및소매업",
"부동산업시설관리지원임대",
"교육서비스업",
"보건업및사회복지서비스업"],
["보건업및사회복지서비스업",
"전기가스수도하수"],
"건설업",
"운수및창고업",
"숙박및음식점업",
"정보통신업",
"금융및보험업",
"전문과학및기술서비스업",
"오락문화및운동관련서비스업",
"기타공공수리및개인서비스업",
"전체",
]
def reformindurstryCode(keycode):
    customKeyCodeMap ={ 
    "농업임업및어업":0,
    "광업":1,
    "제조업":2,
    "전기가스수도하수":3,
    "건설업":4,
    "도매및소매업":5,
    "운수및창고업":6,
    "숙박및음식점업":7,
    "정보통신업":8,
    "금융및보험업":9,
    "부동산업시설관리지원임대":10,
    "전문과학및기술서비스업":11,
    "교육서비스업":12,
    "보건업및사회복지서비스업":13,
    "오락문화및운동관련서비스업":14,
    "기타공공수리및개인서비스업":15,
    "전체":16,
    }
    return customKeyCodeMap[keycode]
    
def reformindurstryColorMap(keycode):
    customColorMap = {
        "농업임업및어업": 'tab:blue',            # 0
        "광업": 'tab:orange',                  # 1
        "제조업": 'tab:green',                 # 2
        "전기가스수도하수": 'tab:red',         # 3
        "건설업": 'tab:purple',                # 4
        "도매및소매업": 'tab:brown',           # 5
        "운수및창고업": 'tab:pink',            # 6
        "숙박및음식점업": 'tab:gray',          # 7
        "정보통신업": 'tab:olive',             # 8
        "금융및보험업": 'tab:cyan',            # 9
        "부동산업시설관리지원임대": 'gold',    # 10
        "전문과학및기술서비스업": 'darkgreen', # 11
        "교육서비스업": 'salmon',              # 12
        "보건업및사회복지서비스업": 'deepskyblue', # 13
        "오락문화및운동관련서비스업": 'darkorange', # 14
        "기타공공수리및개인서비스업": 'mediumvioletred', # 15
        "전체": 'sienna',                      # 16
    }
    return customColorMap

import matplotlib.cm as cm
def drawKMeans():
    inputDataDir = r"resources\dev02\data"
    # 예: 비율 변수들만 선택
    # cols = ["singlePropCompanyRate", "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
    #           "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",]
    cols = ["singlePropCompanyRate",
           "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
              "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",]
    df = loadAllData(inputDataDir, engColumnList)
    typeList = df["industryType"].values
    # print(typeList)
    for i in range(len(typeList)):
        typeList[i] = reformindurstryCode(typeList[i])
    typeList.astype("int32")
    # 정규화 → PCA는 스케일에 민감함
    scaled = StandardScaler().fit_transform(df[cols])

    # PCA 적용 (설명력 높은 주성분만 남김)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    X = pca_result  # (n_samples, 2)

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=6, random_state=0)
    # labels = kmeans.fit_predict(X)  # 각 점의 클러스터 레이블
    # 색 입힌 스캐터 플롯
    plt.scatter(X[:, 0], X[:, 1], c=typeList, cmap="tab20")  # tab10: 범주형 색상
    plt.title("PCA with Cluster Coloring")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()  # 선택: 클러스터 번호 색깔 범례
    plt.show()


if __name__ == "__main__":
    drawKMeans()