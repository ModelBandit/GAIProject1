import copy
import os
import random
import requests
import pandas as pd
import json

encoding = "utf-8"

def openAPI_jsonsave(url):
    # 관측소번호, 시작날짜, 종료날짜 입력 필수
    url = requests.get(url)
    text = url.text

    # JSON 데이터에 액세스
    data = json.loads(text)

    
    df = pd.json_normalize(data)

    # csv 저장
    df.to_csv("sample.csv")

    pass

def erase_etc(string:str):
    string = string.replace(" ","")
    string = string.replace(".","")
    string = string.replace("·","")
    string = string.replace(",","")
    string = string.replace("(","")
    string = string.replace(")","")
    string = string.replace("~","")
    string = string.replace("-","")

    for i in range(10):
        string = string.replace(f"{str(i)}","")
        
    for i in range(65, 91):
        string = string.replace(f"{chr(i)}","")

    return string


# url = "https://kosis.kr/openapi/Param/statisticsParameterData.do?method=getList&apiKey=MjYxOTgyNWE5MTBmMDA5MjRlYzEzZTFkYTFmZmJjZmY=&itmId=T00+T01+T02+T03+T04+T05+T06+T07+T08+T09+T10+T11+T12+T13+T14+T15+T16+T17+T18+T19+T20+T21+&objL1=00+03+04+05+11+21+22+23+24+25+26+29+31+32+33+34+35+36+37+38+39+&objL2=0+&objL3=000+&objL4=&objL5=&objL6=&objL7=&objL8=&format=json&jsonVD=Y&prdSe=F&newEstPrdCnt=3&orgId=101&tblId=DT_1PC1509"
# openAPI_jsonsave(url)

def reposHireData(directory):
    fileNames = os.listdir(directory)
    saveDir = r"D:\GAIP\resources\Preprocess\hireByYear"
    
    InderstryList = ["전체", "A. 농업,임업및어업(01~03)","B. 광업(05~08)","C. 제조업(10~33)","D. 전기,가스,증기및수도사업(35~36)","E. 하수·폐기물처리,원료재생및환경복원업(37~39)","F. 건설업(41~42)","G. 도매및소매업(45~47)","H. 운수업(49~52)","I. 숙박및음식점업(55~56)","J. 출판,영상,방송통신및정보서비스업(58~63)","K. 금융및보험업(64~66)","L. 부동산업및임대업(68~69)","M. 전문,과학및기술서비스업(70~73)","N. 사업시설관리및사업지원서비스업(74~75)","P. 교육서비스업(85)","Q. 보건업및사회복지서비스업(86~87)","R. 예술,스포츠및여가관련서비스업(90~91)","S. 협회및단체,수리및기타개인서비스업(94~96)"]
    for i in range(len(InderstryList)):
        InderstryList[i] = erase_etc(InderstryList[i])

    count = 0
    for f in fileNames:
        path = f"{directory}/{f}"
        originDf = pd.read_csv(path, encoding=encoding)
        columnList = originDf.columns

        dataList = []
        for i in range(2, len(columnList)):
            dataList.append(originDf[columnList[i]])
            pass

        years = len(originDf.index)//11
        for l in range(years):
            newColumList = [ "산업분류", "평균연령", "평균근속년수", "근로일수", "총근로시간수" , "정상근로시간수", "초과근로시간수", "월급여총액", "정액급여", "초과급여", "연간특별급여", "근로자수"]
            df = pd.DataFrame(columns=newColumList)
            for i in range(len(InderstryList)):
                indexList = [InderstryList[i]]
                for j in range(0, len(newColumList)-1):
                    if(years < 3 and j >= 6 and j <= 9): # 원 단위에서 천 단위로 바꿔야함
                        indexList.append(round(float(dataList[i][l*11+j])/1000))
                    else:
                        indexList.append(dataList[i][l*11+j])
                df.loc[i] = indexList
                
            df = df.astype({newColumList[7]:"int32",newColumList[8]:"int32",newColumList[9]:"int32",newColumList[10]:"int32",newColumList[11]:"int32"})
            df.to_csv(f"{saveDir}/{2009+count}.csv", index=False, encoding=encoding)
            count += 1
            


    pass

def reposInderstryData(path, saveDir, columnList, startColumn):
    InderstryList = ["전체", "농업, 임업 및 어업(01~03)","광업(05~08)","제조업(10~34)","전기, 가스, 증기 및 공기조절 공급업(35)","수도, 하수 및 폐기물 처리, 원료 재생업(36~39)","건설업(41~42)","도매 및 소매업(45~47)","운수 및 창고업(49~52)","숙박 및 음식점업(55~56)","정보통신업(58~63)","금융 및 보험업(64~66)","부동산업(68)","전문, 과학 및 기술 서비스업(70~73)","사업시설 관리, 사업 지원 및 임대 서비스업(74~76)","공공행정, 국방 및 사회보장 행정(84)","교육 서비스업(85)","보건업 및 사회복지 서비스업(86~87)","예술, 스포츠 및 여가관련 서비스업(90~91)","협회 및 단체, 수리 및 기타 개인 서비스업(94~96)"]
    for i in range(len(InderstryList)):
        InderstryList[i] = erase_etc(InderstryList[i])

    df = pd.read_csv(path, encoding=encoding)
    df = df[df["행정구역별"] == "전국"]
    dataList = []
    for i in range(startColumn, len(df.columns)):
        dataList.append(df[df.columns[i]])
    
    for i in range(15):
        newColumList = columnList
        indexCount = len(df.index)//(len(newColumList)-1)
        newDf = pd.DataFrame(columns=newColumList)
        for l in range(indexCount):
            newDataList = [InderstryList[l]]
            for j in range(len(columnList)-1):
                if str(dataList[i][l*(len(columnList)-1)+j]).isdigit() == True:
                    # print(str(dataList[i][l*3+j]))
                    newDataList.append(dataList[i][l*(len(columnList)-1)+j])
                else:
                    try:
                        num = int(dataList[i][l*(len(columnList)-1)+j])
                        newDataList.append(num)
                    except:
                        newDataList.append(0)

            newDf.loc[l] = newDataList
        newDf.to_csv(f"{saveDir}/{2006+i}.csv", index=False, encoding=encoding)


    pass

# directory = r"D:\GAIP\resources\Preprocess\hire"
# reposHireData(directory)

# path = r"resources\Preprocess\inderstry\시도산업대표자성별.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\CEOFM"
# columnList = ["산업분류", "사업체수", "남대표", "여대표"]
# reposInderstryData(path, saveDir, columnList, 3)

# path = r"resources\Preprocess\inderstry\시도산업사업체구분별.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\CompanyType"
# columnList = ["산업분류", "사업체수", "종사자수", "단독사업체사업체", "단독사업체종사자", "본사본점등사업체", "본사본점등종사자", "공장지사영업소사업체", "공장지사영업소종사자"]
# reposInderstryData(path, saveDir, columnList, 5)

# path = r"resources\Preprocess\inderstry\시도산업종사상지위별.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\WorkerType"
# columnList = ["산업분류", "종사자수", "자영업자무급가족", "상용종사자", "일용근로자", "기타종사자"]
# reposInderstryData(path, saveDir, columnList, 5)

# path = r"resources\Preprocess\inderstry\시도산업종사자규모.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\WorkerSize"
# columnList = ["산업분류", 
#               "사업체수", "종사자수", 
#               "1명이상5명미만사업체", 
#               "1명이상5명미만종사자", 
#               "5명이상10명미만사업체", 
#               "5명이상10명미만종사자", 
#               "10명이상20명미만사업체", 
#               "10명이상20명미만종사자", 
#               "20명이상50명미만사업체", 
#               "20명이상50명미만종사자", 
#               "50명이상100명미만사업체", 
#               "50명이상100명미만종사자", 
#               "100명이상300명미만사업체", 
#               "100명이상300명미만종사자", 
#               "300명이상500명미만사업체", 
#               "300명이상500명미만종사자", 
#               "500명이상1000명미만사업체", 
#               "500명이상1000명미만종사자", 
#               "1000명이상사업체",
#               "1000명이상종사자"
#               ]
# reposInderstryData(path, saveDir, columnList, 5)

# path = r"resources\Preprocess\inderstry\시도산업종사자성별.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\WorkerFM"
# columnList = ["산업분류", "종사자합", "종사자남", "종사자여"]
# reposInderstryData(path, saveDir, columnList, 3)


def keyCodeConvert(keyCode):
    keyCodeMap = {
    "전체" : "전체",
    '농업임업및어업':"농업임업및어업",
    '광업':"광업",
    '제조업':"제조업",

    '전기가스증기및공기조절공급업':"전기가스수도하수",
    '수도하수및폐기물처리원료재생업':"전기가스수도하수",
    '전기가스증기및수도사업' : "전기가스수도하수", 
    '하수폐기물처리원료재생및환경복원업' : "전기가스수도하수", 

    '건설업':"건설업",
    '도매및소매업':"도매및소매업",
    '운수업':"운수및창고업",
    '운수및창고업':"운수및창고업",
    '숙박및음식점업':"숙박및음식점업",
    '정보통신업':"정보통신업",
    '출판영상방송통신및정보서비스업': "정보통신업",
    '금융및보험업':"금융및보험업",
    '전문과학및기술서비스업':"전문과학및기술서비스업",

    '부동산업':"부동산업시설관리지원임대",
    '부동산업및임대업' : "부동산업시설관리지원임대", 
    '사업시설관리및사업지원서비스업' : "부동산업시설관리지원임대", 
    '사업시설관리사업지원및임대서비스업':"부동산업시설관리지원임대",

    '공공행정국방및사회보장행정':"공공행정국방및사회보장행정", # 스킵

    '교육서비스업':"교육서비스업",
    '보건업및사회복지서비스업':"보건업및사회복지서비스업",
    '예술스포츠및여가관련서비스업':"오락문화및운동관련서비스업",
    '협회및단체수리및기타개인서비스업':"기타공공수리및개인서비스업",
    }
    return keyCodeMap[keyCode]


customKeyCodeList =[ 
"농업임업및어업",
"광업",
"제조업",
"전기가스수도하수",
"건설업",
"도매및소매업",
"운수및창고업",
"숙박및음식점업",
"정보통신업",
"금융및보험업",
"부동산업시설관리지원임대",
"전문과학및기술서비스업",
#"공공행정국방및사회보장행정", # 스킵
"교육서비스업",
"보건업및사회복지서비스업",
"오락문화및운동관련서비스업",
"기타공공수리및개인서비스업",
]
def mergeCSVData1(dirList, saveDir):
    fileNames = os.listdir(dirList[0])
    # 파일 갯수
    for i in range(len(fileNames)-3):
        dfList = []
        for n in dirList:
            path = f"{n}/{2009+i}.csv"
            dfList.append(pd.read_csv(path, encoding=encoding))
        
        for n in range(1, len(dfList)):
            dfList[n] = dfList[n].drop("산업분류", axis=1)
        for n in range(2, len(dfList)):
            try:
                dfList[n] = dfList[n].drop("사업체수", axis=1)
            except:
                pass
            try:
                dfList[n] = dfList[n].drop("종사자수", axis=1)
            except:
                pass
        for n in range(len(dfList)):
            dfList[n] = dfList[n].transpose()
        for n in range(1, len(dfList)):
            dfList[0] = pd.concat([dfList[0],dfList[n]])
        dfList[0] =  dfList[0].transpose()
        dfList[0].to_csv(f"{saveDir}/{2009+i}.csv", index=False, encoding=encoding)

def mergeCSVDataConvertCode(directory, saveDir):
    fileNames = os.listdir(directory)
    # 파일 갯수
    for i in range(len(fileNames)):
        path = f"{directory}/{fileNames[i]}"
        df = pd.read_csv(path, encoding=encoding)
        #4+5 12+14 -15

        for j in range(len(df.index)):
            df["산업분류"][j] = keyCodeConvert(df["산업분류"][j])

        companyCountSizeList = [
            "본사본점등사업체",
            "공장지사영업소사업체",
              ]
        newData = []
        for l in range(len(companyCountSizeList)):
            for j in range(len(df.index)):
                num = 0
                print(df[companyCountSizeList[l]][j])
                num += int(df[companyCountSizeList[l]][j])
                if len(newData) < len(df.index):
                    newData.append(num)
                else:
                    newData[j] += num
        df["다사업사업체"] = newData
        
        workerCountSizeList = [
            "본사본점등종사자",
            "공장지사영업소종사자"
              ]
        newData = []
        for l in range(len(workerCountSizeList)):
            for j in range(len(df.index)):
                num = 0
                print(df[workerCountSizeList[l]][j])
                num += int(df[workerCountSizeList[l]][j])
                if len(newData) < len(df.index):
                    newData.append(num)
                else:
                    newData[j] += num
        df["다사업체종사자"] = newData

        companyCountSizeList = [
              "300명이상500명미만사업체", 
              "500명이상1000명미만사업체", 
              "1000명이상사업체",
              ]
        newData = []
        for l in range(len(companyCountSizeList)):
            for j in range(len(df.index)):
                num = 0
                print(df[companyCountSizeList[l]][j])
                num += int(df[companyCountSizeList[l]][j])
                if len(newData) < len(df.index):
                    newData.append(num)
                else:
                    newData[j] += num
        df["300명이상사업체"] = newData

        workerCountSizeList = [
              "300명이상500명미만종사자", 
              "500명이상1000명미만종사자", 
              "1000명이상종사자"
              ]
        newData = []
        for l in range(len(workerCountSizeList)):
            for j in range(len(df.index)):
                num = 0
                print(df[workerCountSizeList[l]][j])
                num += int(df[workerCountSizeList[l]][j])
                if len(newData) < len(df.index):
                    newData.append(num)
                else:
                    newData[j] += num
        df["300명이상종사자"] = newData
        
        for j in range(1, len(df.columns)):
            print(df.columns[j])
            if(df[df.columns[j]][0] - df[df.columns[j]][15] < 0):
                print(df[df.columns[j]])
                print(df[df.columns[j]][15])
                print(df[df.columns[j]][0])
            df[df.columns[j]][4] += df[df.columns[j]][5]
            df[df.columns[j]][12] += df[df.columns[j]][14]
            df[df.columns[j]][0] -= df[df.columns[j]][15]

        df = df.drop(15, axis=0)
        df = df.drop(14, axis=0)
        df = df.drop(5, axis=0)
        
        for column in companyCountSizeList:
            df = df.drop(column, axis=1)
        for column in workerCountSizeList:
            df = df.drop(column, axis=1)

        companyCountSizeList = [
            "본사본점등사업체",
            "공장지사영업소사업체",
              ]
        workerCountSizeList = [
            "본사본점등종사자",
            "공장지사영업소종사자"
              ]
        
        for column in companyCountSizeList:
            df = df.drop(column, axis=1)
        for column in workerCountSizeList:
            df = df.drop(column, axis=1)
        


        df.to_csv(f"{saveDir}/{2009+i}.csv", index=False, encoding=encoding)
        

            





    
    pass
def numCountChecker(directory):
    fileNames = os.listdir(directory)
    # 파일 갯수
    for i in range(len(fileNames)):
        path = f"{directory}/{fileNames[i]}"
        df = pd.read_csv(path, encoding=encoding)

        for j in range(len(df.index)):
            totalComp = df["사업체수"][j]
            companyCountSizeList = [
                "1명이상5명미만사업체", 
                "5명이상10명미만사업체", 
                "10명이상20명미만사업체", 
                "20명이상50명미만사업체", 
                "50명이상100명미만사업체", 
                "100명이상300명미만사업체", 
                "300명이상사업체",
                ]
            compNum = 0
            for column in companyCountSizeList:
                compNum += int(df[column][j])

            totalWorker = df["종사자합"][j]
            workerCountSizeList = [
                "1명이상5명미만종사자", 
                "5명이상10명미만종사자", 
                "10명이상20명미만종사자", 
                "20명이상50명미만종사자", 
                "50명이상100명미만종사자", 
                "100명이상300명미만종사자", 
                "300명이상종사자",
                ]
            workerNum = 0
            for column in workerCountSizeList:
                workerNum += int(df[column][j])
            if(totalComp != compNum):
                print(f"{fileNames[i]} - {df["산업분류"][j]} - {totalComp}:{compNum}, {totalWorker}:{workerNum}")
                df["300명이상사업체"][j] += (totalComp-compNum)
            if(totalWorker != workerNum):
                print(f"{fileNames[i]} - {df["산업분류"][j]} - {totalComp}:{compNum}, {totalWorker}:{workerNum}")
                df["300명이상종사자"][j] += (totalWorker-workerNum)
            
        df.to_csv(path, index=False, encoding=encoding)

def hireDataConvertCode(directory, saveDir):
    fileNames = os.listdir(directory)
    # 파일 갯수
    for i in range(len(fileNames)):
        path = f"{directory}/{fileNames[i]}"
        df = pd.read_csv(path, encoding=encoding)
        #4+5 12+14 -15

        for j in range(len(df.index)):
            df["산업분류"][j] = keyCodeConvert(df["산업분류"][j])

        for j in range(1, len(df.columns)-1):
            df[df.columns[j]][4] = (df[df.columns[j]][4] * df[df.columns[len(df.columns)-1]][4]) + (df[df.columns[j]][5] * df[df.columns[len(df.columns)-1]][5])
            df[df.columns[j]][12] = (df[df.columns[j]][12] * df[df.columns[len(df.columns)-1]][12]) + (df[df.columns[j]][14] * df[df.columns[len(df.columns)-1]][14])
        
        df[df.columns[len(df.columns)-1]][4] = df[df.columns[len(df.columns)-1]][4] + df[df.columns[len(df.columns)-1]][5]
        df[df.columns[len(df.columns)-1]][12] = df[df.columns[len(df.columns)-1]][12] + df[df.columns[len(df.columns)-1]][14]
        for j in range(1, len(df.columns)-1):
            df[df.columns[j]][4] = round(float(df[df.columns[j]][4] / df[df.columns[len(df.columns)-1]][4]), 1)
            df[df.columns[j]][12] = round(float(df[df.columns[j]][12] / df[df.columns[len(df.columns)-1]][12]), 1)
        
        
        df = df.drop(14, axis=0)
        df = df.drop(5, axis=0)
        


        df.to_csv(f"{saveDir}/{2009+i}.csv", index=False, encoding=encoding)

def convertInderstryRate(directory, saveDir):
    fileNames = os.listdir(directory)

    for i in range(len(fileNames)):
        path = f"{directory}/{fileNames[i]}"
        df = pd.read_csv(path, encoding=encoding)

        companyCountSizeList = [
            "남대표", "여대표",
            "단독사업체사업체","다사업사업체",
            "1명이상5명미만사업체", 
            "5명이상10명미만사업체", 
            "10명이상20명미만사업체", 
            "20명이상50명미만사업체", 
            "50명이상100명미만사업체", 
            "100명이상300명미만사업체", 
            "300명이상사업체",
            ]
        workerCountSizeList = [
            "종사자남", "종사자여",
            "단독사업체종사자","다사업체종사자",
            "자영업자무급가족","상용종사자","일용근로자","기타종사자",
            "1명이상5명미만종사자", 
            "5명이상10명미만종사자", 
            "10명이상20명미만종사자", 
            "20명이상50명미만종사자", 
            "50명이상100명미만종사자", 
            "100명이상300명미만종사자", 
            "300명이상종사자",
            ]
        
        for j in range(len(df.index)):
            for column in companyCountSizeList:
                df[column][j] = round(df[column][j] / df["사업체수"][j], 3)
            for column in workerCountSizeList:
                df[column][j] = round(df[column][j] / df["종사자합"][j], 3)
        

        df.to_csv(f"{saveDir}/{fileNames[i]}", index=False, encoding=encoding)
korColumnList = [
    "산업분류",
    "사업체수","남대표","여대표","단독사업체사업체","다사업사업체",
    "1명이상5명미만사업체","5명이상10명미만사업체","10명이상20명미만사업체","20명이상50명미만사업체",
    "50명이상100명미만사업체","100명이상300명미만사업체","300명이상사업체",
    "종사자합","종사자남","종사자여","단독사업체종사자","다사업체종사자",
    "자영업자무급가족","상용종사자","일용근로자","기타종사자",
    "1명이상5명미만종사자","5명이상10명미만종사자","10명이상20명미만종사자","20명이상50명미만종사자",
    "50명이상100명미만종사자","100명이상300명미만종사자","300명이상종사자",
    "평균연령","평균근속년수","근로일수","총근로시간수","정상근로시간수","초과근로시간수","월급여총액","정액급여","초과급여","연간특별급여"
]
korToEng = {
    "산업분류": "inderstryType",
    "사업체수": "companyCount",
    "남대표": "ownerMaleRate",
    "여대표": "ownerFemaleRate",
    "단독사업체사업체": "singlePropCompanyRate",
    "다사업사업체": "multiBusinessCompanyRate",
    "1명이상5명미만사업체": "U1D5CompanyRate",
    "5명이상10명미만사업체": "U5D10CompanyRate",
    "10명이상20명미만사업체": "U10D20CompanyRate",
    "20명이상50명미만사업체": "U20D50CompanyRate",
    "50명이상100명미만사업체": "U50D100CompanyRate",
    "100명이상300명미만사업체": "U100D300CompanyRate",
    "300명이상사업체": "U300CompanyRate",
    "종사자합": "workerCount",
    "종사자남": "workerMaleRate",
    "종사자여": "workerFemaleRate",
    "단독사업체종사자": "singlePropWorkerRate",
    "다사업체종사자": "multiBusinessWorkerRate",
    "자영업자무급가족": "selfEmpFamilyWorkerRate",
    "상용종사자": "fulltimeWorkerRate",
    "일용근로자": "dayWorkerRate",
    "기타종사자": "etcWorkerRate",
    "1명이상5명미만종사자": "U1D5WorkerRate",
    "5명이상10명미만종사자": "U5D10WorkerRate",
    "10명이상20명미만종사자": "U10D20WorkerRate",
    "20명이상50명미만종사자": "U20D50WorkerRate",
    "50명이상100명미만종사자": "U50D100WorkerRate",
    "100명이상300명미만종사자": "U100D300WorkerRate",
    "300명이상종사자": "U300WorkerRate",
    "평균연령": "avgAge",
    "평균근속년수": "avgServYear",
    "근로일수": "avgWorkDay",
    "총근로시간수": "avgTotalWorkTime",
    "정상근로시간수": "avgRegularWorkDay",
    "초과근로시간수": "avgOverWorkDay",
    "월급여총액": "avgSalary",
    "정액급여": "avgFixedSalary",
    "초과급여": "avgOvertimeSalary",
    "연간특별급여": "avgBonusSalary"
}
def mergeInderstryAndHire(dirList, saveDir):
    fileNames = os.listdir(dirList[0])
    # 파일 갯수
    for i in range(len(fileNames)):
        dfList = []
        for n in dirList:
            path = f"{n}/{2009+i}.csv"
            dfList.append(pd.read_csv(path, encoding=encoding))
        
        for n in range(1, len(dfList)):
            dfList[n] = dfList[n].drop("산업분류", axis=1)
            dfList[n] = dfList[n].drop("근로자수", axis=1)
        for n in range(len(dfList)):
            dfList[n] = dfList[n].transpose()
        for n in range(1, len(dfList)):
            dfList[0] = pd.concat([dfList[0],dfList[n]])
        dfList[0] =  dfList[0].transpose()
        dfList[0] = dfList[0][korColumnList]
        dfList[0] = dfList[0].rename(columns=korToEng)
        dfList[0].to_csv(f"{saveDir}/{2009+i}.csv", index=False, encoding=encoding)

# dirList = [
#     r"resources\Preprocess\inderstry\CEOFM",
#     r"resources\Preprocess\inderstry\WorkerFM",
#     r"resources\Preprocess\inderstry\CompanyType",
#     r"resources\Preprocess\inderstry\WorkerType",
#     r"resources\Preprocess\inderstry\WorkerSize",
# ]
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\total"
# mergeCSVData1(dirList, saveDir)

# directory = r"D:\GAIP\resources\Preprocess\inderstry\total"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\convertCode"
# mergeCSVDataConvertCode(directory, saveDir)

# directory = r"resources\Preprocess\hireByYear"
# saveDir = r"resources\Preprocess\hireYearConvert"
# hireDataConvertCode(directory, saveDir)

# directory = r"resources\Preprocess\inderstry\convertCode"
# numCountChecker(directory)

# directory = r"resources\Preprocess\inderstry\convertCode"
# saveDir = r"resources\Preprocess\inderstry\rate"
# convertInderstryRate(directory, saveDir)
# "avgServYear","avgWorkDay","avgTotalWorkTime","avgRegularWorkDay","avgOverWorkDay","avgSalary","avgFixedSalary","avgOvertimeSalary","avgBonusSalary"] 


def dataSliceAugmentation(inputDir):
    inputNames = os.listdir(inputDir)

    for i in range(len(inputNames)-1):
        iPath = f"{inputDir}/{inputNames[i]}"
        nPath = f"{inputDir}/{inputNames[i+1]}"

        df = pd.read_csv(iPath, encoding=encoding)
        nextDf = pd.read_csv(nPath, encoding=encoding)

        columnList = [*df.columns]

        newDf = pd.DataFrame(columns=df.columns)
        count = 0
        for j in range(len(df.index)):
            index = df.iloc[j]

            columnString = str(index[columnList[0]])
            ndf = nextDf[nextDf[columnList[0]] == columnString]

            dataList = [columnString]
            for l in range(1, len(columnList)):
                a = index[columnList[l]]
                b = ndf[columnList[l]][ndf.first_valid_index()]

                if(str(columnList[l]) == "companyCount" or str(columnList[l]) == "workerCount"):
                    # a = int(a)
                    # b = int(b)
                    dataList.append(round((a + b) / 2, 0))
                else:
                    # a = float(a)
                    # b = float(b)
                    dataList.append(round((a + b) / 2, 3))

            newDf.loc[count] = dataList
            count += 1
        
        savePath = f"{inputDir}/{inputNames[i]}_1"   
        if(os.path.exists(savePath)):
            savePath = f"{inputDir}/{inputNames[i]}_0"   
        newDf.to_csv(savePath, index=False, encoding=encoding)

        


        

if __name__ == "__main__":
    # dirList = [
    #     r"D:\GAIP\resources\Preprocess\inderstry\rate",
    #     r"D:\GAIP\resources\Preprocess\hireYearConvert",
    # ]
    # saveDir = r"resources\dev02\data"
    # mergeInderstryAndHire(dirList, saveDir)

    # # 대표자성별, 종사자성별, 사업체구분 사업체/종사자, 종사자구분 종사자, 종사자규모구분 사업체/종사자, 산업평균
    # # -> 사업체 수, 사업체 대표자 성별 비율, 사업체 사업체 구분 비율, 사업체 종사자규모 구분 비율, 
    # engColumnList = ["inderstryType", 
    #             "companyCount", "ownerMaleRate","ownerFemaleRate", "singlePropCompanyRate", "multiBusinessCompanyRate", 
    #             "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
    #             "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",
    #             "workerCount", "workerMaleRate", "workerFemaleRate", "singlePropWorkerRate", "multiBusinessWorkerRate", 
    #             "selfEmpFamilyWorkerRate", "fulltimeWorkerRate", "dayWorkerRate", "etcWorkerRate",
    #             "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
    #             "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",
    #             "avgAge",
    # inputDir = r"resources\dev02\data"
    # dataSliceAugmentation(inputDir)
    # inputDir = r"resources\dev02\target"
    # dataSliceAugmentation(inputDir)
    pass