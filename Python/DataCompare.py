import copy
import os
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
    with open(path, "r", encoding="utf-8") as f:
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
    with open(path, "r", encoding="utf-8") as f:
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
                
                
    df.to_csv("./resources/Preprocess/TraceTarget.csv", index=False ,encoding="utf-8")

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


def dev02Dataset(directory, dstDir, codeList):
    fileNames = os.listdir(directory)

    dictionary = dict()

    for fileName in fileNames:
        countDictionary = dict()
        print(fileName)
        path = f"{directory}/{fileName}"
        numString = f"p{fileName[6:8]}"
        df = pd.read_csv(path)
        for i in range(df.last_valid_index()):
            code = numString + "0350"
            dfIndex = df.iloc[i]
            try:
                n = int(dfIndex[code])
            except:
                continue



            # 종사자, 남자수, 청년층
            newList = [1,0,0]
            for codetail in codeList:
                code = numString + codetail
                info = dfIndex[code]
                if codetail == "0101" and info == 1:
                    newList[1] = 1
                    
                elif codetail == "0107" and info < 40:
                    newList[2] = 1
            
            code = numString + "0350"
            key = str(int(dfIndex[code]))
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
    with open(f"{dstDir}/Data.json", "w") as f:
        json.dump(dictionary, f, indent=2)

def data_agumentation(directory, dstDir, columnList):
    with open(f"{directory}/Data.json", "r") as f:
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
            for count in range(5, 0, -1): # 직종은 넣었으니 패스
                c = curData[j][count]
                p = preData[j][count]

                if(count >= 5):
                    curData[j].append(int(c) - int(p))
                else:
                    curData[j].insert(count+1 ,int(c) - int(p))

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

        # with open(f"./asd.json","w", encoding="utf-8") as f:
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
        with open(f"./resources/Preprocess/dev02P/workerPercentage/workerPercentage{numString}.json","w", encoding="utf-8") as f:
            json.dump(otherDict, f, indent=2)
        print(f"{numString} 파일 저장")

import matplotlib.pyplot as plt

def show_percentage(dstDir, jobCode, index):
    fileNames = os.listdir(dstDir)
    allList = []
    path = f"{dstDir}/{fileNames[index]}"
    print(path)
    with open(path, "r", encoding="utf-8") as f:
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
    for i in range(1, len(fileNames)):
        prevPath = f"{srcDir}/{fileNames[i-1]}"
        nextPath = f"{srcDir}/{fileNames[i]}"
        prevDf = pd.read_csv(prevPath)[columnList]
        nextDf = pd.read_csv(nextPath)[columnList]


        newList = [prevDf["jobType"][0]]
        for j in columnList:
            if j == "jobType":
                continue

            d = 0
            if int(nextDf[j][0]) > int(prevDf[j][0]):
                d = int(prevDf[j][0]) / int(nextDf[j][0])
                d = 1 - d
                d *= 100
            else:
                d = int(nextDf[j][0]) / int(prevDf[j][0])
                d = d - 1
                d *= 100
            newList.append(f"{d:.6f}")
        print(i-1)
        targetDf.loc[i-1] = newList
        
    targetDf.to_csv("./asd.csv")


    pass

# directory = r"D:\myproject\GAIProject1\resources\dev01\input"
# codeList = ["0101", "0107", "0110", "0350", "1642"]
# refineAgain(directory, codeList)  
# classification_occupation(directory)

# compareCodeList = ["0101", "0107", "0110", "0350"]
# collectCodeList = ["0107", "1642"]
            
# codeList = ["0101","0107"]
# dev02Dataset("./resources/dev01/input", "./resources/Preprocess", codeList)

# columnList = ["jobType", "workerCount", "prevWorkerCount", "maleCount", "prevMaleCount", "femaleCount", "prevFemaleCount", "ageLt40Count", "prevAgeLt40Count", "ageGte40Count", "prevAgeGte40Count"]
# data_agumentation("./resources/Preprocess/dev02P", "./resources/Preprocess/dev02P/AgumentationData", columnList)
# add_salary_min_max("./resources/Preprocess/dev01P/haveJob", "haveJob", "./resources/Preprocess/dev02P/AgumentationData", "AgumentationData")
# workerPercentage("./resources/dev02")
# workerPercentagePlus("./resources/dev02")
# show_percentage(r"D:\myproject\GAIProject1\resources\Preprocess\dev02P\workerPercentage","9", 0)

columnList = ["jobType", "workerCount", "ageGte40Count", "minSalary","maxSalary","meanSalary"]
targetList = ["jobType", "nextWorkerCountRate", "nextAgeGte40Rate","nextMinSalaryRate","nextMaxSalaryRate","nextMeanSalaryRate"]
# targetList = [다음 40대이상 비율, 임금 변화율]

refine_by_columns(f"./resources/dev02", "./", columnList, targetList)