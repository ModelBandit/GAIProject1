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


engColumnList = ["inderstryType", 
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
            #   "companyCount", 
            #   "ownerMaleRate",
            #   "singlePropCompanyRate", 
            #   "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
            #   "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",
            #   "workerCount", 
            #   "workerMaleRate", "singlePropWorkerRate", 
            #   "selfEmpFamilyWorkerRate", "fulltimeWorkerRate", "dayWorkerRate", "etcWorkerRate",
              "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
              "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",
            #   "avgAge",
            "avgServYear",
            "avgWorkDay",
            #   "avgTotalWorkTime",
            #   "avgRegularWorkDay","avgOverWorkDay","avgSalary","avgFixedSalary","avgOvertimeSalary","avgBonusSalary"
            ]

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
    cols = ["singlePropCompanyRate", "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
              "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",]
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
def drawKMeans():
    inputDataDir = r"resources\dev02\data"
    # 예: 비율 변수들만 선택
    cols = ["singlePropCompanyRate", "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
              "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",]
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
        inertias.append(km.inertia_)

    plt.plot(range(1, 21), inertias, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

if __name__ == "__main__":
    drawKMeans()