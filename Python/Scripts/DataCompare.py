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

url = "https://kosis.kr/openapi/Param/statisticsParameterData.do?method=getList&apiKey=MjYxOTgyNWE5MTBmMDA5MjRlYzEzZTFkYTFmZmJjZmY=&itmId=T00+T01+T02+T03+T04+T05+T06+T07+T08+T09+T10+T11+T12+T13+T14+T15+T16+T17+T18+T19+T20+T21+&objL1=00+03+04+05+11+21+22+23+24+25+26+29+31+32+33+34+35+36+37+38+39+&objL2=0+&objL3=000+&objL4=&objL5=&objL6=&objL7=&objL8=&format=json&jsonVD=Y&prdSe=F&newEstPrdCnt=3&orgId=101&tblId=DT_1PC1509"
openAPI_jsonsave(url)
