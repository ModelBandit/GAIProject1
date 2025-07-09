import oracledb as cx_Oracle
import os
import pandas as pd
import numpy as np

projectRoot = "." # D:/GAIP
trainInputDir = "resources/dev02/trainData/inputData"
encoding = "utf-8"
columnList = ["jobType", "workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count","minSalary","maxSalary","meanSalary"]

def getAllDataList():

    dataDir = f"{projectRoot}/{trainInputDir}"
    fileNames = os.listdir(dataDir)

    df = pd.DataFrame(columns=columnList)
    count = 1
    for fName in fileNames:
        path = f"{dataDir}/{fName}"
        newDf = pd.read_csv(path, encoding=encoding)

        newDf["searchCount"] = [count] * len(newDf)
        df = pd.concat([df, newDf])
        count += 1

    return df


class ConnectDB:
    def sql_on(self):
        self.lib_Dir = "C:/instantclient-basic-windows.x64-19.27.0.0.0dbru/instantclient_19_27" # instant clinet 받아서 풀어놓고 처리해야 함.
        cx_Oracle.init_oracle_client(lib_dir=self.lib_Dir)

        self.dsn = cx_Oracle.makedsn("localhost", 1521, sid="xe")
        self.user = "scott"
        self.pwd = "tiger"

        try:
            self.connection = cx_Oracle.connect(user=self.user, password=self.pwd, dsn=self.dsn)
            print("Connected")
        except cx_Oracle.DatabaseError as e:
            print("Fail: ", e)

    def sql_execute(self, queryList):
        if self.connection is None:
            print("ConnectError")
            return
        
        for query in queryList:
            print(query)
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(query)
            except cx_Oracle.DatabaseError as e:
                print("error : ", e)
        
    def sql_off(self):
        self.connection.close()
        pass

    def ImportDataToOracle(self):
        dbColumnList = ["searchCount", "jobType", "workerCount","maleCount","femaleCount","ageLt40Count","ageGte40Count","minSalary","maxSalary","meanSalary"]
        self.sql_on()
        queryList = []
        queryList.append("""CREATE TABLE jobType_statistics(
        searchRound NUMBER(2, 0),
        jobType NUMBER(3, 0),
        workerCount NUMBER(4, 0),
        maleCount NUMBER(4, 0),
        femaleCount NUMBER(4, 0),
        ageLt40Count NUMBER(4, 0),
        ageGte40Count NUMBER(4, 0),
        minSalary NUMBER(4, 0),
        maxSalary NUMBER(4, 0),
        meanSalary NUMBER(6),
        CONSTRAINT id PRIMARY KEY (searchRound, jobType)
        )""")

        df = getAllDataList()
        for i in range(len(df.index)):
            data = df.iloc[i]
            queryList.append(f"INSERT INTO jobType_statistics VALUES ({data[dbColumnList[0]]}, {data[dbColumnList[1]]}, {data[dbColumnList[2]]}, {data[dbColumnList[3]]}, {data[dbColumnList[4]]}, {data[dbColumnList[5]]}, {data[dbColumnList[6]]}, {data[dbColumnList[7]]}, {data[dbColumnList[8]]}, {data[dbColumnList[9]]})")

        queryList.append("commit")
        self.sql_execute(queryList)
        self.sql_off()
        pass

db = ConnectDB()
db.ImportDataToOracle()