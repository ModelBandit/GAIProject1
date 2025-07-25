import copy
import os
import random
import pandas as pd
import json

encoding = "utf-8"


# 해당 폴더 내 csv를 전부 읽어서 성별 나이 학력 직종으로 추적하고 나눠서 json으로 저장함
def trace_target_data(directory, compareCodeList, collectCodeList):
    # 해당 directory안에 파일이름들 읽음
    fileList = os.listdir(directory)
    # data[compareCodeList][List][collectCodeList]
    # dict<List, List<List>>

    # dictionary형로 저장 <str(성별 나이 학력 직종), list[list[회차, 나이, 월급]]
    dictionary = dict()
    
    # 파일마다 반복
    for fileIndex in range(len(fileList)):
        stringNum = f"{fileIndex+1}".zfill(2) # 2자리 숫자형태로 만듬
        df = pd.read_csv(f"{directory}/{fileList[fileIndex]}")
        
        print(f"{fileIndex} - {df.last_valid_index()}")
        for index in range(df.last_valid_index()):
            key = "" # 여기에 str(성별 나이 학력 직종) 형태로 담음
            # if(fileIndex == 19 and index == 4174):
            #     asd = 321
            for collect in compareCodeList:
                column = f"p{stringNum}{collect}"
                if(collect == "0107"): # 나이의 경우
                    col = df[column][index]
                    num = int(col) - fileIndex # 중요 이걸로 첫 관측된 나이 추적함
                    if num < 20: # 20세 이하는 거름
                        num = int(col)
                        
                    key += str(num).zfill(2)
                else:

                    num = int(df[column][index])
                    if(num < 0): # 간혹 -1들어가는거 처리함
                        num = 0
                    key += str(num)

            # if('2-1' in key):
            #     a = 123
            inValue = [] # 이거 안에 [회차, 나이, 월급]
            inValue.append("v"+f"{fileIndex+1}".zfill(2)) # 회차
            for collect in collectCodeList: # 나이 월급. 입력받은 리스트 사용함
                column = f"p{stringNum}{collect}"
                inValue.append(int(df[column][index]))

            outValue = None
            if(key in dictionary): # 이미 넣은게 있는경우. 추적중인 대상은 여기서 다뤄짐
                outValue = dictionary[key]

                if((inValue in outValue) == False): # 동명이인 처리용 (그냥 만나이라 착간한거 일수도 있지만 남겨둠)
                    outValue.append(inValue)
            else: # 새로 넣는 경우
                outValue = []
                outValue.append(inValue)
                dictionary[key] = outValue
    print(dictionary)
    
    dictionary["keys"] = [*dictionary.keys()] # 다 넣고 키값도 저장해둠

    with open("./resources/Preprocess/Datas.json","w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2)


#성별,나이,학력,직군 순으로 합쳐진 str 분리
def keyListup(keyValue):
    dataList = []
    gender = keyValue[0]
    age = keyValue[1:3]
    edu = keyValue[3:4]
    job = keyValue[4:]

    dataList.append(gender)
    dataList.append(age)
    dataList.append(edu)
    dataList.append(job)
    
    # if(int(job) > 999):
    #     for i in dataList:
    #         print(i)
    return dataList

#성별,나이,학력,직군 분리 후 
def keyDecoder(keyValue):
    kList = keyListup(keyValue)
    gender = kList[0]
    age = kList[1]
    edu = kList[2]
    job = kList[3]

    if(gender == "1"):
        gender = "남자"
    else:
        gender = "여자"
    
    if(edu == "9"):
        edu = "박사"
    elif(edu == "8"):
        edu = "석사"
    elif(edu == "7"):
        edu = "학사"
    elif(edu == "6"):
        edu = "전문학사"
    elif(edu == "5"):
        edu = "고등학교"
    elif(edu == "4"):
        edu = "중학교"
    elif(edu == "3"):
        edu = "초등학교"
    elif(edu == "2"):
        edu = "무학"
    elif(edu == "1"):
        edu = "미취학"
    elif(edu == "0"):
        edu = "무응답"


    print(f"성별: {gender}")
    print(f"나이: {age}")
    print(f"최종학력: {edu}")
    print(f"직업 {job} - {jobDecoder(job)}")
    return kList
    
# 코드에 해당하는 직업 찾아냄
def jobDecoder(jobKey):
    jobWord = "판독불가"
    if(int(jobKey) < 0):
        print(jobWord)
        return
    # dir의 파일들 전부확인함 판본별로 번호가 다름 - 다만 5차 번호 선에서 끝나는듯함
    dir = "./resources/Preprocess/jobConvert"
    fileNames = os.listdir(dir)
    jobKey = int(jobKey)

    for f in fileNames:
        path = f"{dir}/{f}"
        df = pd.read_csv(path, encoding=encoding)  

        # df로 읽고 index에서 xxx형태의 코드랑 맞는지 확인함
        c = df["구코드"]
        index = df.index[c == jobKey]
        jobWord = df["구항목명"][index] # 코드에 맞는 문자열 추출
        break

    decodeData = "Error" # 만약 저기서 제대로 처리못하면 Error가 반환됨
    try:
        decodeData = jobWord.iloc(0)[0] # 일반적으로 xxx는 처리됨
    except Exception as e: # xx형태로 입력해야 되는데 xxx형태로 적어서 생기는 문제 처리
        w = str(jobKey)
        if(w[len(w)-1] == '0'):
            decodeData = jobDecoder(w[:len(w)-1]) # 보통 찾는데 없으면 재귀문제가 발생할 수 있긴함
    return decodeData

# 추적한 대상들을 직종코드 별로 분류함
def reclassificate_trace_data(path):
    with open(path, "r", encoding=encoding) as f:
        loaded = json.load(f)

    jobAndCareer = dict()
    
    loadedKeys = loaded["keys"]
    # asd = 0
    for k in loadedKeys:
        # asd += 1
        if(len(loaded[k][0]) < 1): # 자료 없는건 거름
            continue
        infoList = keyListup(k) # 키값들 분리해서 보관
        if((infoList[3] in jobAndCareer.keys()) == True): # 3번은 직종코드 dictionary에 저장된게 있는지 확인함.
            careerList = jobAndCareer[infoList[3]][0]
            careerList.append(loaded[k])
            # keyDecoder(k)
        else:
            tempList = []
            tempList.append(loaded[k])
            jobAndCareer[infoList[3]] = [tempList]
            # keyDecoder(k)
        # print(loaded[k])
    
    keys = [*jobAndCareer.keys()]
    # print(keys)
    jobAndCareer["keys"] = keys
    jobAndCareer["jobs"] = len(keys)


    with open("./resources/Preprocess/jobAndCareer.json","w") as f:
        json.dump(jobAndCareer, f, indent=2)

# 직종코드별로 분류된 대상들 csv로 저장함 
def trace_change(path):
    with open(path, "r", encoding=encoding) as f:
        loaded = json.load(f)
    
    # 회차 리스트
    versionList = []
    for i in range(1,27): 
        versionList.append(f"v{str(i).zfill(2)}")

    # column 작성용
    columnList = ["code", "job"]
    for i in range(1,27): # 01~26회차까지 나이랑 월급
        columnList.append(f"v{str(i).zfill(2)}age")
        columnList.append(f"v{str(i).zfill(2)}salary")

    df = pd.DataFrame(columns=columnList)
    
    keys = loaded["keys"]
    count = 0
    for k in keys: # 키값
        for outValue in loaded[k]: # 바인딩용
            for idx in outValue: # 직종
                # 추적대상
                if len(idx) < 2: # 2개 미만은 뺌. 변화율 확인 불가
                    continue
                newList = []
                
                indexCount = 0 # 26번 돌긴해야되는데 모든 회차에 응답한 사람은 없어서 index 구분해서 처리해야함
                for versionIndex in range(len(versionList)):
                    
                    if len(idx) > indexCount and idx[indexCount][0] == versionList[versionIndex]:
                        age = idx[indexCount][1]
                        salary = idx[indexCount][2]
                        if salary > 0: # 간혹 월급에 -1 적어둔 항목이 나오는거 처리
                            newList.append(idx[indexCount][1]) 
                            newList.append(idx[indexCount][2])
                        else:
                            for l in range(2):
                                newList.append("")  
                        indexCount += 1
                    else:
                        for l in range(2):
                            newList.append("")    

                # 넣기전에 앞쪽에 코드, 직종 형태로 시작하게 추가함
                newList.insert(0, jobDecoder(k))
                newList.insert(0, k)
                df.loc[count] = newList
                count += 1
                
                
    df.to_csv("./resources/Preprocess/TraceTarget.csv", index=False ,encoding=encoding)

# 이전 회차와 현재 회차 비교하여 임금 변화율 체크함.
def rate_of_change(path):
    df = pd.read_csv(path)
    tag = "salary"
    versionList = []
    for i in range(2,27):# 이전꺼랑 비교하니 2부터 시작
        versionList.append(f"v{str(i).zfill(2)}")
    versionList.insert(0, "code")

    outDF = pd.DataFrame(columns=versionList)
    count = 0

    for i in range(df.last_valid_index()): # df의 인덱스 수만큼 반복
        if(df.iloc[i].notnull().sum() <= 4): # 코드, 직종 [나이 월급] 1개 면 총 4개 -> 4개 이하 = 데이터 1개 = 거름
            continue

        curIndex = df.iloc[i]
        newIndex = [curIndex["code"]]
        for j in range(2,27):
            preNumString = str(j-1).zfill(2)
            curNumString = str(j).zfill(2)

            preCol = f"v{preNumString}{tag}"
            curCol = f"v{curNumString}{tag}"
            
            # 2개만 있으면 가장 앞쪽이랑 나중이랑도 체크하게 하려했는데 복잡해보여서 취소함
            # if(str(curIndex[preCol]).isdigit() == False):
            #     for l in range(26,j,-1):
            #         preNumString = str(l).zfill(2)
            #         preCol = f"v{preNumString}{tag}"
            #         if(str(curIndex[preCol]).isdigit() == True):
            #             break

            # if(str(curIndex[curCol]).isdigit() == False):
            #     for l in range(j, 27):
            #         curNumString = str(l).zfill(2)
            #         curCol = f"v{curNumString}{tag}"
            #         if(str(curIndex[curCol]).isdigit() == True):
            #             break

            # 숫자로 변형되는지 확인하던 파트. df 자료형이라 그런지 안되길래 try로 우회함
            # if(str(curIndex[preCol]).isdigit() == False or str(curIndex[curCol]).isdigit() == False):
            #     print(int(curIndex[preCol]))
            #     print(curIndex[curCol])
            #     continue
            try:
                preSalary = int(curIndex[preCol])
                curSalary = int(curIndex[curCol])
            except: # 형변환 실패하면 보통 공백임
                newIndex.append("")
                continue
            
            per = curSalary / preSalary
            value = ""
            if per >= 1.0:
                per = (per - 1.0)*100
            else:
                per = (1.0 - per)*-100
            value += str(f"{per:.2f}")
            newIndex.append(value)

        # print(newIndex)
        # print(outDF)            
        outDF.loc[count] = newIndex
        count += 1

    outDF.to_csv("./resources/Preprocess/rateOfChange.csv", index=False, encoding=encoding)
    pass

def dev01Dataset():
    # export_csv_Part_Data("D:/project/AI-Project/resources/compareData", "D:/project/AI-Project/resources", "haveJob", "p", compareCodeList)

    # trace_target_data(r"D:\project\AI-Project\resources\Preprocess\haveJob", compareCodeList, collectCodeList)
    # reclassificate_trace_data("D:/project/AI-Project/resources/Preprocess/Datas.json")
    # trace_change("./resources/Preprocess/jobAndCareer.json")

    # rate_of_change("./resources/Preprocess/TraceTarget.csv")
    pass


def dev02Dataset(directory, dstPath, codeList):
    fileNames = os.listdir(directory)

    dictionary = dict()

    for fileName in fileNames:
        countDictionary = dict()
        print(fileName)
        path = f"{directory}/{fileName}"
        numString = f"p{fileName[6:8]}"
        df = pd.read_csv(path)
        for i in range(df.last_valid_index()):
            code = "0350"
            dfIndex = df.iloc[i]
            try:
                n = int(dfIndex[code])
            except:
                continue



            # 종사자, 남자수, 청년층
            newList = [1,0,0]
            for codetail in codeList:
                code = codetail
                info = dfIndex[code]
                if codetail == "0101" and info == 1:
                    newList[1] = 1
                    
                elif codetail == "0107" and info < 40:
                    newList[2] = 1
            
            code = "0350"
            key = str(int(dfIndex[code]))
            
            # if newList[2] < 1: # 새로하기 그래서 수동으로 하는중
            #     continue
            if(key in countDictionary.keys()):
                for di in range(len(newList)):

                    countDictionary[key][di] += newList[di]
            else:
                countDictionary[key] = newList
        countDictionary["keys"] = [*countDictionary.keys()]
        countDictionary["count"] = len(countDictionary["keys"])
        dictionary[numString] = countDictionary

    dictionary["keys"] = [*dictionary.keys()]
    dictionary["count"] = len(dictionary["keys"])
    # with open(f"{dstDir}/Data.json", "w") as f:
    #     json.dump(dictionary, f, indent=2)
    with open(f"{dstPath}", "w") as f:
        json.dump(dictionary, f, indent=2)

def data_agumentation(directory, dstDir, columnList):
    with open(f"{directory}/Data.json", "r") as f: # 파일이름 수동조정됨
        loaded = json.load(f)

    outkeys = loaded["keys"] # 회차
    # count = loaded[outkeys]["count"]

    for outkeyIndex in range(len(outkeys)):
        df = pd.DataFrame(columns=columnList)
        inkeys = loaded[outkeys[outkeyIndex]]["keys"] # 직종

        preVersionDF = None
        if(outkeyIndex <= 0):
            preVersionDF = copy.deepcopy(loaded[outkeys[outkeyIndex]])
        else:
            preVersionDF = copy.deepcopy(loaded[outkeys[outkeyIndex-1]])
        curVersionDF = copy.deepcopy(loaded[outkeys[outkeyIndex]])

        # 먼저 [직종(키), 직종별 총 종사자, 남자, 40세 미만]을 뽑아서 정리함
        curData = []
        preData = []
        for inkeyIndex in range(len(inkeys)):

            cJobTypeList = curVersionDF[inkeys[inkeyIndex]]
            
            if(inkeys[inkeyIndex] in preVersionDF.keys()):
                pJobTypeList = []
                pJobTypeList = preVersionDF[inkeys[inkeyIndex]]
            else:
                pJobTypeList = []
                pJobTypeList = copy.deepcopy(cJobTypeList)

            cJobTypeList.insert(0, inkeys[inkeyIndex])

            if(cJobTypeList == pJobTypeList):
                pJobTypeList = copy.deepcopy(cJobTypeList)
            else:
                pJobTypeList.insert(0, inkeys[inkeyIndex])

            # 이후 여자, 40세 이상을 뽑아서 추가함
            cJobTypeList.insert(3, int(cJobTypeList[1]) - int(cJobTypeList[2]))
            pJobTypeList.insert(3, int(pJobTypeList[1]) - int(pJobTypeList[2]))
            cJobTypeList.append(int(cJobTypeList[1]) - int(cJobTypeList[4]))
            pJobTypeList.append(int(pJobTypeList[1]) - int(pJobTypeList[4]))

            curData.append(cJobTypeList)
            preData.append(pJobTypeList)

        for j in range(len(preData)):
            # 직종, 종사자 수, (작년 종사자 수), 남, (작년 남),  여, (작년 여), 40세 미만, (작년 40미만), 40세 이상, (작년 40 이상)
            # for count in range(5, 0, -1): # 직종은 넣었으니 패스
            #     c = curData[j][count]
            #     p = preData[j][count]

            #     if(count >= 5):
            #         curData[j].append(int(c) - int(p))
            #     else:
            #         curData[j].insert(count+1 ,int(c) - int(p))

            info = curData[j]
            df.loc[j] = info
        numString = str(outkeyIndex+1).zfill(2)
        df.to_csv(f"{dstDir}/AgumentationData{numString}p.csv", index=False, encoding=encoding)

def add_salary_min_max(srcDir, srcFileTemplate, dstDir, dstFileTemplate):
    
    for i in range(1,27):
        numString = str(i).zfill(2)
        srcPath = f"{srcDir}/{srcFileTemplate}{numString}p.csv"
        srcDf = pd.read_csv(srcPath)
        
        dstPath = f"{dstDir}/{dstFileTemplate}{numString}p.csv"
        dstDf = pd.read_csv(dstPath)

        minSalarys = []
        maxSalarys = []
        meanSalarys = []
        jobTypes = dstDf["jobType"]
        for t in jobTypes:
            ageTag = f"p{numString}0107"
            # srcDf = srcDf[srcDf[ageTag] < 40] # 수동 조정됨

            tag = f"p{numString}0350"
            salarys = srcDf[srcDf[tag] == t]
            
            sTag = f"p{numString}1642"
            jobSalarys = salarys[salarys[sTag] > 0]
            filtJobSalarys = jobSalarys[sTag]
            minV = 0
            maxV = 0
            mean = 0
            if(len(jobSalarys) > 1):
                minV = min(filtJobSalarys)
                maxV = max(filtJobSalarys)
                mean = filtJobSalarys.mean()

            minSalarys.append(int(minV))
            maxSalarys.append(int(maxV))
            meanSalarys.append(f"{mean:.1f}")
        
        dstDf["minSalary"] = minSalarys
        dstDf["maxSalary"] = maxSalarys
        dstDf["meanSalary"] = meanSalarys
        dstDf.to_csv(dstPath, index=False, encoding=encoding)


    pass

def workerPercentage(dstDir):
    fileNames = os.listdir(dstDir)

    for i in range(0,len(fileNames)):
        path = f"{dstDir}/{fileNames[i]}"
        df = pd.read_csv(path)

        WorkerKind = dict()

        jobTypes = df["jobType"]

        workerCount = df["workerCount"]
        allWorker = 0

        male = df["maleCount"]
        maleWorkersCount = 0

        female = df["femaleCount"]
        femaleWorkersCount = 0

        for j in range(len(jobTypes)):
            jt = str(jobTypes[j])
            c = jt[0]
            if(c == "1"):
                if(len(jt) < 3 or jt[2] != "0"):
                    c = "2"
            if(c == "1"):
                print(f"{jt} - {jobDecoder(jt)}")

            if (c in WorkerKind.keys()):
                WorkerKind[c][0] += int(workerCount[j])
                WorkerKind[c][1] += int(male[j])
                WorkerKind[c][2] += int(female[j])
            else:
                WorkerKind[c] = [int(workerCount[j]), int(male[j]), int(female[j])]
            allWorker += int(workerCount[j])
            maleWorkersCount += int(male[j])
            femaleWorkersCount += int(female[j])
            

        WorkerKind["all"] = allWorker
        for j in WorkerKind.keys():
            if(j == "all"):
                break
            # print(j)
            fw = WorkerKind[j][2]
            fper = fw / femaleWorkersCount
            WorkerKind[j].append(fper)

            mw = WorkerKind[j][1]
            mper = mw / maleWorkersCount
            WorkerKind[j].insert(2, mper)

            workers = WorkerKind[j][0]
            per = workers / allWorker
            WorkerKind[j].insert(1, per)


        numString = f"{i+1}".zfill(2)
        keyList = sorted(WorkerKind.keys())
        otherDict = dict()
        for j in keyList:
            otherDict[j] = WorkerKind[j]
        
        print(*otherDict.keys())
        with open(f"./resources/Preprocess/dev02P/workerPercentage/workerPercentage{numString}.json","w", encoding="utf-8") as f:
            json.dump(otherDict, f, indent=2)
        print(f"{numString} 파일 저장")


def refineAgain(directory, codeList):
    fileNames = os.listdir(directory)
    for n in range(len(fileNames)):
        path = f"{directory}/{fileNames[n]}"
        df = pd.read_csv(path)
        tempDF = pd.DataFrame()

        for i in codeList:
            code = f"p{str(n+1).zfill(2)}{i}"
            tempDF[i] = df[code]
        tempDF.to_csv(f"refine{str(n+1).zfill(2)}.csv", index=False)
        
    pass

def classification_occupation(directory):
    fileNames = os.listdir(directory)
    with open(r"D:\myproject\GAIProject1\resources\Preprocess\dev01P\jobCode.json", "r") as f:
        jobCode = json.load(f)
    for n in range(len(fileNames)):
        path = f"{directory}/{fileNames[n]}"
        df = pd.read_csv(path)

        for jn in jobCode["keys"]:
            tempDF = df[df["0350"] == float(jn)]
            if tempDF.index.size < 1:
                continue

            for column in tempDF.columns:
                tempDF.rename(columns = {column : str(column).zfill(4)})
            d = f"./resources/dev02/byOccupation/{jn}"
            if not os.path.exists(d):
                os.mkdir(d)

            tempDF.to_csv(f"{d}/{str(n+1).zfill(2)}.csv", index=False)
        
    pass

def workerPercentagePlus(dstDir):
    fileNames = os.listdir(dstDir)

    for i in range(0,len(fileNames)):
        path = f"{dstDir}/{fileNames[i]}"
        df = pd.read_csv(path)

        WorkerKind = dict()

        jobTypes = df["jobType"]

        workerCount = df["workerCount"]
        allWorker = 0

        male = df["maleCount"]
        maleWorkersCount = 0

        female = df["femaleCount"]
        femaleWorkersCount = 0

        for j in range(len(jobTypes)):
            jt = str(jobTypes[j])
            c = jt[0]
            if(c == "1"):
                if(len(jt) < 3 or jt[2] != "0"):
                    c = "2"
            if(c == "1"):
                print(f"{jt} - {jobDecoder(jt)}")


            if (c in WorkerKind.keys() and jt in WorkerKind[c].keys()):
                WorkerKind[c][jt][0] += int(workerCount[j])
                WorkerKind[c][jt][1] += int(male[j])
                WorkerKind[c][jt][2] += int(female[j])
            else:
                # print(jt)
                if c not in WorkerKind.keys():
                    WorkerKind[c] = dict()
                WorkerKind[c][jt] = [int(workerCount[j]), int(male[j]), int(female[j])]
            allWorker += int(workerCount[j])
            maleWorkersCount += int(male[j])
            femaleWorkersCount += int(female[j])
            

        for j in WorkerKind.keys():
            jobAll = 0
            jobMale = 0
            jobFemale = 0
            for l in WorkerKind[j].keys():
                jobAll += WorkerKind[j][l][0]
                jobMale += WorkerKind[j][l][1]
                jobFemale += WorkerKind[j][l][2]
            WorkerKind[j]["info"] = [jobAll, jobMale, jobFemale]

        
        for j in WorkerKind.keys():
            for l in WorkerKind[j].keys():
                if(l == "info"):
                    break
                fw = WorkerKind[j][l][2]
                allfw = WorkerKind[j]["info"][2]
                if(allfw > 1):
                    fper = fw / allfw
                    WorkerKind[j][l].append(round(fper,6))
                else:
                    WorkerKind[j][l].append(0)

                mw = WorkerKind[j][l][1]
                allmw = WorkerKind[j]["info"][1]
                if(allmw > 1):
                    mper = mw / allmw
                    WorkerKind[j][l].insert(2, round(mper,6))
                else:
                    WorkerKind[j][l].insert(2, 0)

                workers = WorkerKind[j][l][0]
                allw = WorkerKind[j]["info"][0]
                if(allw > 1):
                    per = workers / allw
                    WorkerKind[j][l].insert(1, round(per,6))
                else:
                    WorkerKind[j][l].insert(2, 1)

        # with open(f"./asd.json","w", encoding=encoding) as f:
        #     json.dump(WorkerKind, f, indent=2)

            fw = WorkerKind[j]["info"][2]
            fper = fw / femaleWorkersCount
            WorkerKind[j]["info"].append(round(fper,6))

            mw = WorkerKind[j]["info"][1]
            mper = mw / maleWorkersCount
            WorkerKind[j]["info"].insert(2, round(mper,6))

            workers = WorkerKind[j]["info"][0]
            per = workers / allWorker
            WorkerKind[j]["info"].insert(1, round(per,6))

        WorkerKind["all"] = [allWorker, maleWorkersCount, round(maleWorkersCount/allWorker, 6), femaleWorkersCount, round(femaleWorkersCount/allWorker, 6)]


        numString = f"{i+1}".zfill(2)
        keyList = sorted(WorkerKind.keys())
        otherDict = dict()
        for j in keyList:
            if(type(WorkerKind[j]) == type(dict)):
                keyListmore = sorted(WorkerKind[j].keys())
                otherDict[j] = dict()
                for l in keyListmore:
                    otherDict[j][l] = WorkerKind[j][l]
            otherDict[j] = WorkerKind[j]
        
        print(*otherDict.keys())
        with open(f"./resources/Preprocess/dev02P/workerPercentage/workerPercentage{numString}.json","w", encoding=encoding) as f:
            json.dump(otherDict, f, indent=2)
        print(f"{numString} 파일 저장")

import matplotlib.pyplot as plt

def show_percentage(dstDir, jobCode, index):
    fileNames = os.listdir(dstDir)
    allList = []
    path = f"{dstDir}/{fileNames[index]}"
    print(path)
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)
        
    dataDict = data[jobCode]
    keys = [*dataDict.keys()]

    ratioList = []
    labelsList = []
    for j in range(len(keys)):
        if keys[j] == "info":
            continue
            
        d = dataDict[keys[j]][0]
        ratioList.append(d)
        labelsList.append(keys[j])
    allList.append([ratioList, labelsList])
    
    plt.pie(allList[index][0], labels=allList[index][1], autopct='%.1f%%')
    plt.show()

def refine_by_columns(srcDir, dstDir, columnList, targetList):
    fileNames = os.listdir(srcDir)

    targetDf = pd.DataFrame(columns=targetList)
    for i in range(0, len(fileNames)-1):
        curPath = f"{srcDir}/{fileNames[i]}"
        nextPath = f"{srcDir}/{fileNames[i+1]}"
        curDf = pd.read_csv(curPath)[columnList]
        nextDf = pd.read_csv(nextPath)[columnList]

        for l in range(curDf.last_valid_index()):
            nextL = 0
            curData = curDf["jobType"].values[l]
            nextDataList = nextDf["jobType"].values
            if((curData in nextDataList) == True):
                while(curData != nextDataList[nextL]):
                    nextL += 1
            
            newList = [curDf["jobType"][l]]
            for j in columnList:
                d = 0
                if j == "jobType":
                    continue
                
                
                nextData = int(nextDf[j][nextL])
                curData = int(curDf[j][l])
                
                if curData == 0:
                    newList.append(f"{0}")
                    continue

                if j == "ageGte40Count":
                    nextworkerData = int(nextDf["workerCount"][nextL])
                    curworkerData = int(curDf["workerCount"][l])

                    d = (nextData/nextworkerData) / (curData/curworkerData) 
                else:
                    d = nextData / curData

                if d > 1:
                    d = d - 1
                    d *= 100
                else:
                        d = 1 - d
                        d *= 100
                newList.append(f"{d:.2f}")
            targetDf.loc[l] = newList
        numString = str(i+1).zfill(2)
        targetDf.to_csv(f"{dstDir}/{numString}.csv", index=False)


    pass

# directory = r"D:\myproject\GAIProject1\resources\dev01\input"
# codeList = ["0101", "0107", "0110", "0350", "1642"]
# refineAgain(directory, codeList)  
# classification_occupation(directory)

# compareCodeList = ["0101", "0107", "0110", "0350"]
# collectCodeList = ["0107", "1642"]
            
# codeList = ["0101","0107"]
# dev02Dataset("./resources/input", "./resources/Preprocess/Data.json", codeList)

# columnList = ["jobType", "workerCount", "prevWorkerCount", "maleCount", "prevMaleCount", "femaleCount", "prevFemaleCount", "ageLt40Count", "prevAgeLt40Count", "ageGte40Count", "prevAgeGte40Count"]

# columnList = ["jobType", "workerCount", "maleCount", "femaleCount", "ageLt40Count", "ageGte40Count"]
# data_agumentation("./resources/Preprocess", "./resources/dev02/normal/inputData", columnList)

# add_salary_min_max("./resources/Preprocess/dev01P/haveJob", "haveJob", "./resources/dev02/normal/inputData", "AgumentationData")
# workerPercentage("./resources/dev02")
# workerPercentagePlus("./resources/dev02")
# show_percentage(r"D:\myproject\GAIProject1\resources\Preprocess\dev02P\workerPercentage","9", 0)

columnList = ["jobType", "workerCount", "ageGte40Count", "minSalary","maxSalary","meanSalary"]
targetList = ["jobType", "nextWorkerCountRate", "nextAgeGte40Rate","nextMinSalaryRate","nextMaxSalaryRate","nextMeanSalaryRate"]
# refine_by_columns(f"./resources/dev02/normal/inputData", f"./resources/dev02/normal/targetData", columnList, targetList)


# 사용전에 "수동" 이라고 검색해서 수정할 것
def fakeSourceData_under():
    codeList = ["0101","0107"]
    dev02Dataset("./resources/input", "./resources/Preprocess/underData.json", codeList)

    columnList = ["jobType", "workerCount", "maleCount", "femaleCount", "ageLt40Count", "ageGte40Count"]
    data_agumentation("./resources/Preprocess", "./resources/dev02/40under/inputData", columnList)
    add_salary_min_max("./resources/Preprocess/dev01P/haveJob", "haveJob", "./resources/dev02/40under/inputData", "AgumentationData")
    
    columnList = ["jobType", "workerCount", "ageGte40Count", "minSalary","maxSalary","meanSalary"]
    targetList = ["jobType", "nextWorkerCountRate", "nextAgeGte40Rate","nextMinSalaryRate","nextMaxSalaryRate","nextMeanSalaryRate"]
    refine_by_columns(f"./resources/dev02/40under/inputData", "./resources/dev02/40under/targetData", columnList, targetList)

# 안쓰기로 함
def fakeSourceData_on():
    codeList = ["0101","0107"]
    dev02Dataset("./resources/input", "./resources/Preprocess/onData.json", codeList)

    columnList = ["jobType", "workerCount", "maleCount", "femaleCount", "ageLt40Count", "ageGte40Count"]
    data_agumentation("./resources/Preprocess", "./resources/dev02/40on/inputData", columnList)
    add_salary_min_max("./resources/Preprocess/dev01P/haveJob", "haveJob", "./resources/dev02/40on/inputData", "AgumentationData")
    
    columnList = ["jobType", "workerCount", "ageGte40Count", "minSalary","maxSalary","meanSalary"]
    targetList = ["jobType", "nextWorkerCountRate", "nextAgeGte40Rate","nextMinSalaryRate","nextMaxSalaryRate","nextMeanSalaryRate"]
    refine_by_columns(f"./resources/dev02/40on/inputData", "./resources/dev02/40on/targetData", columnList, targetList)

# 안쓰기로 함
def buildFakeInput(srcDir, dstDir):
    dataKind = ["inputData"]
    
    for folderName in dataKind:
        youngDataDir = f"{srcDir}/40under/{folderName}"
        oldDataDir = f"{srcDir}/40on/{folderName}"
        dataDir = f"{srcDir}/normal/{folderName}"

        fileNames = os.listdir(dataDir)

        count = 1
        for fn in fileNames:
            fdd = f"{dstDir}/count{str(count).zfill(2)}"
            if not os.path.exists(fdd):
                os.makedirs(fdd)

            youngDataPath = f"{youngDataDir}/{fn}"
            oldDataPath = f"{oldDataDir}/{fn}"
            dataPath = f"{dataDir}/{fn}"

            df = pd.read_csv(dataPath, encoding=encoding)
            youngDf = pd.read_csv(youngDataPath, encoding=encoding)
            oldDf = pd.read_csv(oldDataPath, encoding=encoding)

            for i in range(df.last_valid_index()+1):

                curJobType = int(df["jobType"][i])
                curYoungDf = youngDf[youngDf["jobType"] == curJobType]
                curOldDf = oldDf[oldDf["jobType"] == curJobType]
                if (len(curYoungDf.index) < 1 or len(curOldDf.index) < 1):
                    continue

                fakeDataDir = f"{fdd}/{folderName}"
                if not os.path.exists(fakeDataDir):
                    os.mkdir(fakeDataDir)
                fakeDataPath = f"{fakeDataDir}/{df["jobType"][i]}.csv"

                columnList = df.columns
                # print(columnList)

                newDf = pd.DataFrame(columns=columnList)

                newList = []
                newList.append(curYoungDf.iloc[0])
                newList.append(curOldDf.iloc[0])
                newList.append(df.iloc[i])

                youngCount = int(df[columnList[4]][i])
                oldCount = int(df[columnList[5]][i])
                if youngCount < 1 or oldCount < 1:
                    continue

                youngMaleCount = int(curYoungDf[columnList[2]])
                youngMalePer = youngMaleCount / youngCount

                oldMaleCount = int(curOldDf[columnList[2]])
                oldMalePer = oldMaleCount / oldCount

                for j in range(1, youngCount):
                    newOldDf = copy.deepcopy(curOldDf)
                    per = (j+1) / youngCount
                    # 종사자
                    newOldDf[columnList[1]] += j
                    # 다른 나이대
                    newOldDf[columnList[4]] += j
                    
                    #성별
                    newMale = 0
                    newFemale = 0
                    for l in range(j):
                        if(newMale < youngMaleCount and newFemale < (youngCount - youngMaleCount)):
                            realNum = random.random() % 1
                            if(realNum <= oldMalePer):
                                newOldDf[columnList[2]] += 1
                                newMale += 1
                            else:
                                newOldDf[columnList[3]] += 1
                                newFemale += 1

                        elif(newMale < youngMaleCount):
                            newOldDf[columnList[2]] += 1
                            newMale += 1

                        else:
                            newOldDf[columnList[3]] += 1
                            newFemale += 1

                    # 임금
                    newOldDf[columnList[6]].iloc[0] = newOldDf[columnList[6]].iloc[0] + float(int(curYoungDf[columnList[6]].iloc[0]) - int(newOldDf[columnList[6]].iloc[0])) * per
                    newOldDf[columnList[7]].iloc[0] = newOldDf[columnList[7]].iloc[0] + float(int(curYoungDf[columnList[7]].iloc[0]) - int(newOldDf[columnList[7]].iloc[0])) * per
                    newOldDf[columnList[8]].iloc[0] = newOldDf[columnList[8]].iloc[0] + float(int(curYoungDf[columnList[8]].iloc[0]) - int(newOldDf[columnList[8]].iloc[0])) * per
                    newList.append(newOldDf.iloc[0])
                    
                for j in range(1, oldCount):
                    newYoungDf = copy.deepcopy(curYoungDf)
                    per = (j+1) / oldCount
                    # 종사자
                    newYoungDf[columnList[1]] += (j)
                    # 다른 나이대
                    newYoungDf[columnList[5]] += (j)
                    
                    #성별
                    newMale = 0
                    newFemale = 0
                    for l in range(j):
                        if(newMale < oldMaleCount and newFemale < (oldCount - oldMaleCount)):
                            realNum = random.random() % 1
                            if(realNum <= youngMalePer):
                                newYoungDf[columnList[2]] += 1
                                newMale += 1
                            else:
                                newYoungDf[columnList[3]] += 1
                                newFemale += 1

                        elif(newMale < youngMaleCount):
                            newYoungDf[columnList[2]] += 1
                            newMale += 1

                        else:
                            newYoungDf[columnList[3]] += 1
                            newFemale += 1

                    # 임금
                    newYoungDf[columnList[6]] += (int(curOldDf[columnList[6]]) - newYoungDf[columnList[6]]) * per
                    newYoungDf[columnList[7]] += (int(curOldDf[columnList[7]]) - newYoungDf[columnList[7]]) * per
                    newYoungDf[columnList[8]] += (int(curOldDf[columnList[8]]) - newYoungDf[columnList[8]]) * per
                    newList.append(newYoungDf.iloc[0])
                
                for d in range(len(newList)):
                    newDf.loc[d] = newList[d]
                    s1 = str(df["jobType"][i])
                newDf.to_csv(f"{fakeDataPath}", index=False, encoding=encoding)
            count+=1

# 안쓰기로 함
def buildFakeTarget(srcDir, dstDir):
    folderName = "targetData"
    youngDataDir = f"{srcDir}/40under/{folderName}"
    oldDataDir = f"{srcDir}/40on/{folderName}"
    dataDir = f"{srcDir}/normal/{folderName}"

    fileNames = os.listdir(dataDir)

    count = 1
    for fn in fileNames:
        fdd = f"{dstDir}/count{str(count).zfill(2)}"
        if not os.path.exists(fdd):
            os.makedirs(fdd)

        youngDataPath = f"{youngDataDir}/{fn}"
        oldDataPath = f"{oldDataDir}/{fn}"
        dataPath = f"{dataDir}/{fn}"

        df = pd.read_csv(dataPath, encoding=encoding)
        youngDf = pd.read_csv(youngDataPath, encoding=encoding)
        oldDf = pd.read_csv(oldDataPath, encoding=encoding)

        for i in range(df.last_valid_index()+1):

            curJobType = int(df["jobType"][i])
            curYoungDf = youngDf[youngDf["jobType"] == curJobType]
            curOldDf = oldDf[oldDf["jobType"] == curJobType]
            if (len(curYoungDf.index) < 1 or len(curOldDf.index) < 1):
                continue

            fakeDataDir = f"{fdd}/{folderName}"
            if not os.path.exists(fakeDataDir):
                os.mkdir(fakeDataDir)
            fakeDataPath = f"{fakeDataDir}/{df["jobType"][i]}.csv"

            columnList = df.columns
            # print(columnList)

            newDf = pd.DataFrame(columns=columnList)

            newList = []
            newList.append(curYoungDf.iloc[0])
            newList.append(curOldDf.iloc[0])
            newList.append(df.iloc[i])

            youngCount = int(df[columnList[4]][i])
            oldCount = int(df[columnList[5]][i])
            if youngCount < 1 or oldCount < 1:
                continue

            youngMaleCount = int(curYoungDf[columnList[2]])
            youngMalePer = youngMaleCount / youngCount

            oldMaleCount = int(curOldDf[columnList[2]])
            oldMalePer = oldMaleCount / oldCount

            for j in range(1, youngCount):
                newOldDf = copy.deepcopy(curOldDf)
                per = (j+1) / youngCount
                # 종사자
                newOldDf[columnList[1]] += j
                # 다른 나이대
                newOldDf[columnList[4]] += j
                
                #성별
                newMale = 0
                newFemale = 0
                for l in range(j):
                    if(newMale < youngMaleCount and newFemale < (youngCount - youngMaleCount)):
                        realNum = random.random() % 1
                        if(realNum <= oldMalePer):
                            newOldDf[columnList[2]] += 1
                            newMale += 1
                        else:
                            newOldDf[columnList[3]] += 1
                            newFemale += 1

                    elif(newMale < youngMaleCount):
                        newOldDf[columnList[2]] += 1
                        newMale += 1

                    else:
                        newOldDf[columnList[3]] += 1
                        newFemale += 1

                # 임금
                newOldDf[columnList[6]].iloc[0] = newOldDf[columnList[6]].iloc[0] + float(int(curYoungDf[columnList[6]].iloc[0]) - int(newOldDf[columnList[6]].iloc[0])) * per
                newOldDf[columnList[7]].iloc[0] = newOldDf[columnList[7]].iloc[0] + float(int(curYoungDf[columnList[7]].iloc[0]) - int(newOldDf[columnList[7]].iloc[0])) * per
                newOldDf[columnList[8]].iloc[0] = newOldDf[columnList[8]].iloc[0] + float(int(curYoungDf[columnList[8]].iloc[0]) - int(newOldDf[columnList[8]].iloc[0])) * per
                newList.append(newOldDf.iloc[0])
                
            for j in range(1, oldCount):
                newYoungDf = copy.deepcopy(curYoungDf)
                per = (j+1) / oldCount
                # 종사자
                newYoungDf[columnList[1]] += (j)
                # 다른 나이대
                newYoungDf[columnList[5]] += (j)
                
                #성별
                newMale = 0
                newFemale = 0
                for l in range(j):
                    if(newMale < oldMaleCount and newFemale < (oldCount - oldMaleCount)):
                        realNum = random.random() % 1
                        if(realNum <= youngMalePer):
                            newYoungDf[columnList[2]] += 1
                            newMale += 1
                        else:
                            newYoungDf[columnList[3]] += 1
                            newFemale += 1

                    elif(newMale < youngMaleCount):
                        newYoungDf[columnList[2]] += 1
                        newMale += 1

                    else:
                        newYoungDf[columnList[3]] += 1
                        newFemale += 1

                # 임금
                newYoungDf[columnList[6]] += (int(curOldDf[columnList[6]]) - newYoungDf[columnList[6]]) * per
                newYoungDf[columnList[7]] += (int(curOldDf[columnList[7]]) - newYoungDf[columnList[7]]) * per
                newYoungDf[columnList[8]] += (int(curOldDf[columnList[8]]) - newYoungDf[columnList[8]]) * per
                newList.append(newYoungDf.iloc[0])
            
            for d in range(len(newList)):
                newDf.loc[d] = newList[d]
                s1 = str(df["jobType"][i])
            newDf.to_csv(f"{fakeDataPath}", index=False, encoding=encoding)
        count+=1

# buildFakeInput("./resources/dev02", "./resources/dev02/fake")

def CheckDataShape(srcDir, dstDir):
    srcNames = os.listdir(srcDir)
    dstNames = os.listdir(dstDir)

    for i in range(len(srcNames)):
        print(srcNames[i])
        srcPath = f"{srcDir}/{srcNames[i]}"
        dstPath = f"{dstDir}/{dstNames[i]}"

        srcDf = pd.read_csv(srcPath, encoding=encoding)
        print(srcDf.shape)
        dstDf = pd.read_csv(dstPath, encoding=encoding)
        print(dstDf.shape)

        srcJobType = srcDf["jobType"]
        dstJobType = dstDf["jobType"]

        srcLen = len(srcJobType.index)
        dstLen = len(dstJobType.index)
        if(srcLen > dstLen):
            for j in range(srcLen):
                try:
                    if(srcJobType.iloc[j] == dstJobType.iloc[j]) == False:
                        print(srcJobType.iloc[j])
                except:
                    print(srcJobType.iloc[j])
                    pass
        elif(srcLen < dstLen):
            for j in range(dstLen):
                try:
                    if(srcJobType.iloc[j] == dstJobType.iloc[j]) == False:
                        print(dstJobType.iloc[j])
                except:
                    print(dstJobType.iloc[j])
                    pass
        else:
            print(f"{srcNames[i]} - 정상")
            

def absoluteValueTarget(iputDir, saveDir):
    inputNames = os.listdir(iputDir)
    columnList = ["jobType","workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count","minSalary","maxSalary","meanSalary"]

    for i in range(len(inputNames)-1):
        print(inputNames[i])
        srcPath = f"{srcDir}/{inputNames[i]}"
        dstPath = f"{srcDir}/{inputNames[i+1]}"

        srcDf = pd.read_csv(srcPath, encoding=encoding)
        print(srcDf.shape)
        dstDf = pd.read_csv(dstPath, encoding=encoding)
        print(dstDf.shape)

        newDf1 = pd.DataFrame(columns=srcDf.columns)
        newDf2 = pd.DataFrame(columns=srcDf.columns)
        
        for j in range(len(srcDf.index)):
            jobType = srcDf["jobType"][j]
            targetDf = dstDf[dstDf["jobType"] == jobType]
            if(targetDf.size > 0):
                if((int(targetDf.iloc[0][columnList[6]]) > 0 or int(targetDf.iloc[0][columnList[7]]) > 0 or int(targetDf.iloc[0][columnList[8]]) > 0)):
                    newDf1.loc[j] = targetDf.iloc[0]
                    newDf2.loc[j] = srcDf.iloc[j]

        newDf1 = newDf1.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"int32"})
        newDf2 = newDf2.astype({columnList[0]:"int32",columnList[1]:"int32",columnList[2]:"int32",columnList[3]:"int32",columnList[4]:"int32",columnList[5]:"int32",columnList[6]:"int32",columnList[7]:"int32"})
        # print(newDf1.dtypes)
        numString = str(i+1).zfill(2)
        newDf1.to_csv(f"{saveDir}/targetData/{numString}.csv", index=False, encoding=encoding)
        newDf2.to_csv(f"{saveDir}/inputData/{numString}.csv", index=False, encoding=encoding)



def merge_jobCode_industrialCode(srcDir, dstDir):
    inputNames = os.listdir(srcDir)
    codeColumn = "jobType"

    codeDir = r"D:\GAIP\resources\Preprocess\jobConvert"
    codeFiles = os.listdir(codeDir)
    jobCode5 = pd.read_csv(f"{codeDir}/{codeFiles[0]}")
    jobCode6 = pd.read_csv(f"{codeDir}/{codeFiles[1]}")
    jobCode7 = pd.read_csv(f"{codeDir}/{codeFiles[2]}")

    # 1 10 1차 11 20 2차 21 7차 차이

    typeList = ["code","workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count", "minSalary","maxSalary","meanSalary", "jobCount"]
    for i in range(len(inputNames)):
        srcPath = f"{srcDir}/{inputNames[i]}"

        srcDf = pd.read_csv(srcPath, encoding=encoding)
        srcDf["jobCount"] = 1
        
        if i < 10:
            jobCodeDf = jobCode5
            print("jobCode5")
        elif i < 20:
            jobCodeDf = jobCode6
        else:
            jobCodeDf = jobCode7

        indurstryDf = pd.DataFrame(columns=typeList)
        count = 0

        for j in range(len(srcDf.index)):
            if j == 5:
                pass
            srcType = srcDf["jobType"][j]
            # print(srcType)


            data = jobCodeDf[jobCodeDf["code"] == srcType]
            if(len(data.index) < 1):
                if(len(str(srcType)) >= 3 and str(srcType)[2] == "0"):
                    srcType = int(str(srcType)[0:2])
                cl = []
                jobCodeDf = jobCode5
                d = jobCodeDf[jobCodeDf["code"] == srcType]
                if(len(d.index) > 0):
                    print("jobCode5")
                    cl.append(d)

                jobCodeDf = jobCode6
                d = jobCodeDf[jobCodeDf["code"] == srcType]
                if(len(d.index) > 0):
                    print("jobCode6")
                    cl.append(d)

                jobCodeDf = jobCode7
                d = jobCodeDf[jobCodeDf["code"] == srcType]
                if(len(d.index) > 0):
                    print("jobCode7")
                    cl.append(d)

                if(len(cl) < 1):
                    indurstryData = indurstryDf[indurstryDf["code"] == "기타"]
                    if(len(indurstryData.index) > 0):
                        addList = ["workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count", "minSalary","maxSalary","meanSalary", "jobCount"]
                        indurstryData = indurstryDf[indurstryDf["code"] == "기타"]
                        for column in addList:
                            if int(srcDf[column][j]) < 1:
                                continue
                            indurstryData[column] += srcDf[column][j]
                            
                    else:
                        indurstryDf.loc[count] = srcDf.iloc[j]
                        indurstryDf["code"][count] = "기타"
                    continue
                else:
                    data = cl[0]
        
            srcType = str(data.values[0][2])
            srcType = srcType.replace(" ", "")
            srcType = srcType.replace(",", "")
            if(srcType == "기타"):
                pass
            # print(srcType)
            srcDf.iloc[j]["code"] = srcType
            indurstryData = indurstryDf[indurstryDf["code"] == srcType]
            if(len(indurstryData.index) > 0):
                # print(indurstryData)
                addList = ["workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count", "minSalary","maxSalary","meanSalary", "jobCount"]
                for column in addList:
                    if int(srcDf[column][j]) < 1:
                        continue
                    indurstryData[column] += srcDf[column][j]
                    # try:
                    # except:
                    #     a = indurstryData[column]
                    #     b = srcDf[column][j]
                    #     a += b
            else:
                indurstryDf.loc[count] = srcDf.iloc[j]
                indurstryDf["code"][count] = srcType
            count += 1

            salaryList = ["minSalary","maxSalary","meanSalary"]
            for salary in salaryList:
                indurstryData[salary] = indurstryData[salary] / srcDf["jobCount"][j]


        indurstryDf = indurstryDf.astype({typeList[1]:"int32",typeList[2]:"int32",typeList[3]:"int32",typeList[4]:"int32",typeList[5]:"int32",typeList[6]:"int32",typeList[7]:"int32",typeList[8]:"float64",typeList[9]:"int32"})
        indurstryDf.to_csv(f"{dstDir}/{str(i+1).zfill(2)}.csv", index=False, encoding=encoding)

# 데이터 부족하면 안합치는 버전으로
def jobCode_industrialCode(srcDir, dstDir):
    inputNames = os.listdir(srcDir)
    codeColumn = "jobType"

    codeDir = r"D:\GAIP\resources\Preprocess\jobConvert"
    codeFiles = os.listdir(codeDir)
    jobCode5 = pd.read_csv(f"{codeDir}/{codeFiles[0]}")
    jobCode6 = pd.read_csv(f"{codeDir}/{codeFiles[1]}")
    jobCode7 = pd.read_csv(f"{codeDir}/{codeFiles[2]}")

    # 1 10 1차 11 20 2차 21 7차 차이

    typeList = ["code","workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count"]
    for i in range(len(inputNames)):
        srcPath = f"{srcDir}/{inputNames[i]}"

        srcDf = pd.read_csv(srcPath, encoding=encoding)
        
        
        if i < 10:
            jobCodeDf = jobCode5
            print("jobCode5")
        elif i < 20:
            jobCodeDf = jobCode6
        else:
            jobCodeDf = jobCode7

        indurstryDf = pd.DataFrame(columns=typeList)
        count = 0
        for j in range(len(srcDf.index)):
            if j == 5:
                pass
            srcType = srcDf["jobType"][j]
            # print(srcType)


            data = jobCodeDf[jobCodeDf["code"] == srcType]
            if(len(data.index) < 1):
                if(len(str(srcType)) >= 3 and str(srcType)[2] == "0"):
                    srcType = int(str(srcType)[0:2])
                cl = []
                jobCodeDf = jobCode5
                d = jobCodeDf[jobCodeDf["code"] == srcType]
                if(len(d.index) > 0):
                    print("jobCode5")
                    cl.append(d)

                jobCodeDf = jobCode6
                d = jobCodeDf[jobCodeDf["code"] == srcType]
                if(len(d.index) > 0):
                    print("jobCode6")
                    cl.append(d)

                jobCodeDf = jobCode7
                d = jobCodeDf[jobCodeDf["code"] == srcType]
                if(len(d.index) > 0):
                    print("jobCode7")
                    cl.append(d)

                if(len(cl) < 1):
                    indurstryDf.loc[count] = srcDf.iloc[j]
                    indurstryDf["code"][count] = "기타"
                    continue
                else:
                    data = cl[0]
        
            srcType = str(data.values[0][2])
            srcType = srcType.replace(" ", "")
            srcType = srcType.replace(",", "")
            if(srcType == "기타"):
                pass
            # print(srcType)
            srcDf.iloc[j]["code"] = srcType
            indurstryData = indurstryDf[indurstryDf["code"] == srcType]
            indurstryDf.loc[count] = srcDf.iloc[j]
            indurstryDf["code"][count] = srcType
            count += 1
        indurstryDf = indurstryDf.astype({typeList[1]:"int32",typeList[2]:"int32",typeList[3]:"int32",typeList[4]:"int32",typeList[5]:"int32"})
        indurstryDf.to_csv(f"{dstDir}/{str(i+1).zfill(2)}.csv", index=False, encoding=encoding)



# srcDir = "resources/dev02/originData/inputData"
# dstDir = r"D:\GAIP\resources\dev02\haveSalary\inputData"
# # absoluteValueTarget(srcDir, dstDir)
# merge_jobCode_industrialCode(srcDir, dstDir)

# jobCode_industrialCode(srcDir, dstDir)

def Indurstrial_target_maker(srcDir, dstDir, columnList):
    companyTarget = "./resources/company_Industrial.csv"
    workerTarget = "./resources/Worker_Industrial.csv"
    dataColumnList = ["Industrial", *range(2006,2021)]
    fileNames = os.listdir(srcDir)


    companyDf = pd.read_csv(companyTarget)
    workerDf = pd.read_csv(workerTarget)

    for i in range(len(fileNames)):
        path = f"{srcDir}/{fileNames[i]}"
        print(i)
        targetList = [dataColumnList[0], dataColumnList[i+1]]
        originCodeList = copy.deepcopy([*companyDf["Industrial"].values])

        inputDf = pd.read_csv(path, encoding=encoding)
        targetDf = pd.DataFrame(columns=columnList)

        
        # mCompDF1 = companyDf[targetList]
        # mWorkDF1 = workerDf[targetList]

        etcNumList = []
        for j in range(len(inputDf.index)):
            targetIndex = []
            codeString = str(inputDf["code"][j])
            if codeString == "기타":
                etcNumList.append(j)
                continue
            
            if(codeString in originCodeList):
                originCodeList.remove(codeString)
            targetIndex.append(codeString)

            mCompDF2 = companyDf[companyDf[targetList[0]] == codeString]
            print(mCompDF2.values[0][i+1])
            targetIndex.append(mCompDF2.values[0][i+1])

            mWorkDF2 = workerDf[workerDf[targetList[0]] == codeString]
            print(mWorkDF2.values[0][i+1])
            targetIndex.append(mWorkDF2.values[0][i+1])
            targetDf.loc[j] = targetIndex

        targetIndex = []
        targetIndex.append("기타")
        targetIndex.append(0)
        targetIndex.append(0)
        mCompDF2 = None
        mWorkDF2 = None
        
        for j in range(len(originCodeList)):
            mCompDF2 = companyDf[companyDf[targetList[0]] == originCodeList[j]]
            mWorkDF2 = workerDf[workerDf[targetList[0]] == originCodeList[j]]
            # print(mCompDF2.values[0][i+1])
            targetIndex[1] += mCompDF2.values[0][i+1]
            targetIndex[2] += mWorkDF2.values[0][i+1]

            # print(mWorkDF2.values[0][i+1])

        # 추가하는 자리
        for j in range(len(etcNumList)):
            listDF = pd.DataFrame([targetIndex], columns=columnList)
            targetDf = pd.concat([targetDf[:etcNumList[j]], listDF, targetDf[etcNumList[j]:]])

        targetDf.to_csv(f"{dstDir}/{str(i+8).zfill(2)}.csv", index=False, encoding=encoding)

    pass
# srcDir = "resources/dev02/haveSalary/inputData"
# dstDir = "resources/dev02/haveSalary/targetData"
# columnList = ["code","companyCount","workerCount"]
# Indurstrial_target_maker(srcDir, dstDir, columnList)

# def resave(path):
#     df = pd.read_csv(path, encoding=encoding)
#     for j in range(len(df.index)):
#         dfString = str(df["Industrial"][j])
#         dfString = dfString.replace(" ", "")
#         dfString = dfString.replace(",", "")
#         df["Industrial"][j] = dfString
#     df.to_csv(path, index=False, encoding=encoding)
# companyTarget = r"D:\GAIP\resources\company_Industrial.csv"
# workerTarget = r"D:\GAIP\resources\Worker_Industrial.csv"

def dataSliceAugmentation(inputDir, targetDir):
    inputNames = os.listdir(inputDir)

    for i in range(len(inputNames)-1):
        iPath = f"{inputDir}/{inputNames[i]}"
        nPath = f"{inputDir}/{inputNames[i+1]}"

        df = pd.read_csv(iPath, encoding=encoding)
        nextDf = pd.read_csv(nPath, encoding=encoding)

        columnList = [*df.columns]

        etcSkipList = []
        newDf = pd.DataFrame(columns=df.columns)
        count = 0
        for j in range(len(df.index)):
            index = df.iloc[j]

            columnString = str(index[columnList[0]])
            ndf = nextDf[nextDf[columnList[0]] == columnString]

            if len(ndf.index) < 1:
                etcSkipList.append(columnString)
                continue

            dataList = [columnString]
            for l in range(1, len(columnList)):
                a = index[columnList[l]]
                b = ndf[columnList[l]][ndf.first_valid_index()]

                print(a)
                print(b)

                dataList.append((a + b) // 2)

            newDf.loc[count] = dataList
            count += 1
        
        for etc in  etcSkipList:
            ndf = df[df[columnList[0]] == etc]
            for l in range(1, len(columnList)):
                ldata = newDf[newDf[columnList[0]] == "기타"][columnList[l]]
                ldata += ndf[columnList[l]]
            for l in range(1, len(columnList)-1):
                newDf[newDf[columnList[0]] == "기타"][columnList[l]] = newDf[newDf[columnList[0]] == "기타"][columnList[l]] // len(etcSkipList)
        
        savePath = f"{inputDir}/half_{inputNames[i]}"    
        newDf.to_csv(savePath, index=False, encoding=encoding)

        
    targetNames = os.listdir(targetDir)
    for i in range(len(inputNames)-1):
        iPath = f"{targetDir}/{targetNames[i]}"
        nPath = f"{targetDir}/{targetNames[i+1]}"

        df = pd.read_csv(iPath, encoding=encoding)
        nextDf = pd.read_csv(nPath, encoding=encoding)

        columnList = [*df.columns]

        etcSkipList = []
        newDf = pd.DataFrame(columns=df.columns)
        count = 0
        for j in range(len(df.index)):
            index = df.iloc[j]

            columnString = str(index[columnList[0]])
            ndf = nextDf[nextDf[columnList[0]] == columnString]

            if len(ndf.index) < 1:
                etcSkipList.append(columnString)
                continue

            dataList = [columnString]
            for l in range(1, len(columnList)):
                dataList.append((index[columnList[l]] + ndf[columnList[l]][ndf.first_valid_index()]) // 2)

            newDf.loc[count] = dataList
            count += 1
        
        for etc in  etcSkipList:
            ndf = df[df[columnList[0]] == etc]
            for l in range(1, len(columnList)):
                ldata = newDf[newDf[columnList[0]] == "기타"][columnList[l]]
                ldata += ndf[columnList[l]]
            for l in range(1, len(columnList)-1):
                newDf[newDf[columnList[0]] == "기타"][columnList[l]] = newDf[newDf[columnList[0]] == "기타"][columnList[l]] // len(etcSkipList)
        
        savePath = f"{targetDir}/half_{targetNames[i]}"    
        newDf.to_csv(savePath, index=False, encoding=encoding)

    
    pass

# inputDir = "resources/dev02/haveSalary/inputData"
# targetDir = "resources/dev02/haveSalary/targetData"
# dataSliceAugmentation(inputDir, targetDir)