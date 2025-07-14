import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

from ML import loadAllData, DecisionTreeML
from sklearn.model_selection import train_test_split

projectRoot = "./"
resourcesDir = "resources/dev02/haveSalary/예외"
encoding = "utf-8-sig"
# columnList = ["maleCount","femaleCount","ageLt40Count","ageGte40Count"]
columnList = ["code", "workerCount", "maleCount","femaleCount","ageLt40Count","ageGte40Count", "minSalary", "maxSalary","meanSalary"]
targetColumnList = ["workerCount"] # "companyCount",
codeList = [
    "농업임업및어업",
    "광업",
    "제조업",
    "전기가스증기및공기조절공급업",
    "수도하수및폐기물처리원료재생업",
    "건설업",
    "도매및소매업",
    "운수및창고업",
    "숙박및음식점업",
    "정보통신업",
    "금융및보험업",
    "부동산업",
    "전문과학및술서비스업",
    "사업시설관리사업지원및임대서비스업",
    "공공행정국방및사회보장행정",
    "교육서비스업",
    "보건업및사회복지서비스업",
    "예술스포츠및여가관련서비스업",
    "협회및단체수리및기타개인서비스업",
    "기타"
]

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def loadAllMerge(directory, targetList):
    df = loadAllData(directory, columnList)

    newDf = pd.DataFrame(columns=targetList)
    count = 0
    for i in range(0, len(codeList)):
        dd = df[df[columnList[0]] == codeList[i]]

        if len(dd.index) < 1:
            continue
        
        newList = [codeList[i], 0,0,0,0,0,0,0,0]
        for l in range(1, len(targetList)-3):
            
            data = dd[targetList[l]].values
            for j in range(len(dd.index)):
                # print(data)
                newList[l] += int(data[j])
        
        for l in range(6, len(targetList)):
            data = dd[targetList[l]].values
            n = len(dd.index)
            for j in range(n):
                # print(data)
                newList[l] += int(data[j])
            newList[l] = newList[l] // n

        newDf.loc[count] = newList
        count+=1
    
    return newDf

def sumDF(dataFrame:pd.DataFrame):
    num = 0
    for i in range(len(dataFrame.index)):
        num += dataFrame.iloc[i]
    return num

def heatmap():
    inputDataDir = r".\resources\dev02\haveSalary\inputData"
    inputDf = loadAllData(inputDataDir, columnList)
    
    corr_matrix = pd.DataFrame(inputDf, columns=inputDf.columns).corr()
    plt.figure(figsize=(200,200))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()

def columnRate():
    # targetList = [columnList[0], columnList[5]]
    df = loadAllMerge(resourcesDir, columnList)

    for i in codeList:
        newdf = df[df[columnList[0]] == i]
        if len(newdf.index) < 1:
            continue
        labels = [columnList[4], columnList[5]]
        ratio = [sumDF(newdf[columnList[4]]),sumDF(newdf[columnList[5]])]
        print(ratio)
        plt.title(i)
        plt.pie(ratio, labels=labels, autopct='%.1f%%')
        plt.show()

def jobAndPay():
    df = loadAllMerge(resourcesDir, columnList)
    categories =[*df[columnList[0]]]

    # 각 카테고리에 대해 바 3개씩 (예: 세 가지 모델의 점수)
    values1 = [*df[columnList[6]]]
    values2 = [*df[columnList[7]]]
    values3 = [*df[columnList[8]]]

    x = np.arange(len(categories))  # x축 위치 [0, 1, 2, ..., 19]
    width = 0.1  # 바 폭

    
    plt.bar(x - width, values1, width, label='min')
    plt.bar(x,         values2, width, label='max')
    plt.bar(x + width, values3, width, label='mean')
    plt.xticks(x, categories, rotation=45)  # x축 이름 및 회전
    plt.legend()
    plt.tight_layout()
    plt.show()

from sklearn.tree import plot_tree
def showTree():
    inputDataDir = r".\resources\dev02\haveSalary\inputData"
    targetDataDir = r".\resources\dev02\haveSalary\targetData"
    inputDf = loadAllData(inputDataDir, columnList)
    targetDf = loadAllData(targetDataDir, targetColumnList)
    trainInput, testInput, trainTarget, testTarget = train_test_split(inputDf, targetDf, test_size=0.2, random_state=42)
    tree = DecisionTreeML(trainInput, testInput, trainTarget, testTarget)

    plt.figure(figsize=(10,7))
    plot_tree(tree)
    plt.show()

# columnRate()
# jobAndPay()
showTree()


