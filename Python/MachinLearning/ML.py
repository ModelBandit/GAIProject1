import pandas as pd
import os

from sklearn.tree import DecisionTreeRegressor

global projectRoot
global dataDir
global directory
global fileNames

projectRoot = "D:/myproject/GAIProject1"
dataDir = "resources/dev02"
directory = None
fileNames = None

def set_project_root(rootDir):
    global projectRoot
    projectRoot = rootDir
    build_data_path()

def build_data_path():
    global dataDir
    global directory
    global fileNames
    directory = f"{projectRoot}/{dataDir}"
    fileNames = os.listdir(directory)

set_project_root("D:/myproject/GAIProject1")


import matplotlib.pyplot as plt

def visualize(keyindex):
    dataDict = dict()
    selectColumnList = ["workerCount","minSalary","maxSalary","meanSalary"]

    for n in fileNames:
        path = f"{directory}/{n}"
        # print(path)
        
        jobtypes = pd.read_csv(path)["jobType"]
        df = pd.read_csv(path)[selectColumnList]
        
        for j in range(len(jobtypes)):
            t = str(jobtypes[j])
            if(t in dataDict.keys()):
                dataDict[t].append(df.iloc[j])
                # print(len(dataDict[t]))
            else:
                dataDict[t] = [df.iloc[j]]
                # print(dataDict[t])

    # for i in dataDict.keys():
        # print(f"{i} - {len(dataDict[i])}")
        # print(dataDict[i])

    keys = [*dataDict.keys()]
    k = keys[keyindex]
    d = dataDict[k]
    plt.title(k)
    countList = []
    minList = []
    maxList = []
    meanList = []

    for i in range(len(d)):
        count = i
        mis = d[i]["minSalary"]
        mas = d[i]["maxSalary"]
        mean = d[i]["meanSalary"]
        countList.append(count)
        minList.append(mis)
        maxList.append(mas)
        meanList.append(mean)
    plt.scatter(countList,minList,c="red")
    plt.scatter(countList,maxList,c="blue")
    plt.scatter(countList,meanList,c="yellow")

    plt.show()

import numpy as np

def ml():
    dataDict = dict()
    selectColumnList = ["workerCount","minSalary","maxSalary","meanSalary"]

    for n in fileNames:
        path = f"{directory}/{n}"
        # print(path)
        
        jobtypes = pd.read_csv(path)["jobType"]
        df = pd.read_csv(path)[selectColumnList]
        
        for j in range(len(jobtypes)):
            t = str(jobtypes[j])
            if(t in dataDict.keys()):
                dataDict[t].append(df.iloc[j])
                # print(len(dataDict[t]))
            else:
                dataDict[t] = [df.iloc[j]]
                # print(dataDict[t])

    # for i in dataDict.keys():
        # print(f"{i} - {len(dataDict[i])}")
        # print(dataDict[i])

    keys = [*dataDict.keys()]
    k = keys[keyindex]
    d = dataDict[k]
    plt.title(k)
    countList = []
    minList = []
    maxList = []
    meanList = []

    for i in range(len(d)):
        count = i
        mis = d[i]["minSalary"]
        mas = d[i]["maxSalary"]
        mean = d[i]["meanSalary"]
        countList.append(count)
        minList.append(mis)
        maxList.append(mas)
        meanList.append(mean)

    data = np.array([])