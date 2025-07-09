import pandas as pd
import os

from sklearn.linear_model import LinearRegression

global projectRoot
global inputDir
global targetDir

projectRoot = "D:/GAIP"
trainInputDir = "resources/dev02/trainData/inputData"
trainTargetDir = "resources/dev02/trainData/targetData"
testInputDir = "resources/dev02/testData/inputData"
testTargetDir = "resources/dev02/testData/targetData"
encoding = "utf-8"

def loadAllData(directory, columnList):
    fileNames = os.listdir(directory)

    df = pd.DataFrame(columns=columnList)
    for fName in fileNames:
        path = f"{directory}/{fName}"
        newDf = pd.read_csv(path, encoding=encoding)[columnList]

        df = pd.concat([df, newDf])
    return df


def ml():
    inputColumnList = ["workerCount","ageLt40Count","ageGte40Count","minSalary","maxSalary","meanSalary"]
    targetColumnList = ["nextWorkerCountRate","nextAgeGte40Rate","nextMinSalaryRate","nextMaxSalaryRate","nextMeanSalaryRate"]

    inputDataDir = f"{projectRoot}/{trainInputDir}"
    targetDataDir = f"{projectRoot}/{trainTargetDir}"

    inputData = loadAllData(inputDataDir, inputColumnList).values
    targetData = loadAllData(targetDataDir, targetColumnList).values
    
    print(inputData.shape)
    print(targetData.shape)
    pass
    lr = LinearRegression()
    lr.fit(inputData, targetData)

    print(lr.score())
    # print(inputData)



ml()
