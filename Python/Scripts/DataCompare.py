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
    string = string.replace(",","")
    string = string.replace("(","")
    string = string.replace(")","")
    string = string.replace("~","")
    string = string.replace("-","")

    for i in range(10):
        string = string.replace(f"{str(i)}","")
    return string


# url = "https://kosis.kr/openapi/Param/statisticsParameterData.do?method=getList&apiKey=MjYxOTgyNWE5MTBmMDA5MjRlYzEzZTFkYTFmZmJjZmY=&itmId=T00+T01+T02+T03+T04+T05+T06+T07+T08+T09+T10+T11+T12+T13+T14+T15+T16+T17+T18+T19+T20+T21+&objL1=00+03+04+05+11+21+22+23+24+25+26+29+31+32+33+34+35+36+37+38+39+&objL2=0+&objL3=000+&objL4=&objL5=&objL6=&objL7=&objL8=&format=json&jsonVD=Y&prdSe=F&newEstPrdCnt=3&orgId=101&tblId=DT_1PC1509"
# openAPI_jsonsave(url)

def reposHireData(directory, columnList):
    fileNames = os.listdir(directory)
    
    InderstryList = ["전체", "농업 수렵업 및 임업(01-02)","어 업(05)","광 업(10-12)","제 조 업(15-37)","전기, 가스 및 수도사업(40-41)","건설업(45-46)","도매 및 소매업(50∼52)","숙박 및 음식점업(55)","운수업(60-63)","통신업(64)","금융 및 보험업(65-67)","부동산 및 임대업(70-71)","사업 서비스업(72-75)","교육서비스업(80)","보건 및 사회복지사업(85-86)","오락, 문화 및 운동관련서비스업(87-88)","기타 공공, 수리 및 개인서비스업(90-93)"]
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
            df.to_csv(f"./{2006+count}.csv", index=False, encoding=encoding)
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
# path = r"resources\Preprocess\inderstry\시도산업대표자성별.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\CEOFM"
# columnList = ["산업분류", "합", "남", "여"]
# reposInderstryData(path, saveDir, columnList, 3)

# path = r"resources\Preprocess\inderstry\시도산업사업체구분별.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\CompanyType"
# columnList = ["산업분류", "사업체 전체", "종사자 전체", "단독사업체 사업체", "단독사업체 종사자", "본사,본점 등 사업체", "본사,본점 등 종사자", "공장,지사(점),영업소 사업체", "공장,지사(점),영업소 종사자"]
# reposInderstryData(path, saveDir, columnList, 5)

# path = r"resources\Preprocess\inderstry\시도산업종사상지위별.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\WorkerType"
# columnList = ["산업분류", "종사자 수", "자영업자무급가족", "상용종사자", "일용근로자", "기타종사자"]
# reposInderstryData(path, saveDir, columnList, 5)

# path = r"resources\Preprocess\inderstry\시도산업종사자규모.csv"
# saveDir = r"D:\GAIP\resources\Preprocess\inderstry\WorkerSize"
# columnList = ["산업분류", 
#               "사업체 수", "종사자 수", 
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
#               "300명이상1000명미만사업체", 
#               "300명이상500명미만종사자", 
#               "500명이상500명미만사업체", 
#               "500명이상1000명미만종사자", 
#               "1000명이상사업체",
#               "1000명이상종사자"
#               ]
# reposInderstryData(path, saveDir, columnList, 5)

path = r"resources\Preprocess\inderstry\시도산업종사자성별.csv"
saveDir = r"D:\GAIP\resources\Preprocess\inderstry\WorkerFM"
columnList = ["산업분류", "종사자 합", "남", "여"]
reposInderstryData(path, saveDir, columnList, 3)