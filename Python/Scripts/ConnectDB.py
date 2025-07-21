import oracledb as cx_Oracle
import os
import pandas as pd

projectRoot = "." # D:/GAIP
trainInputDir = r"resources\predictRate"
encoding = "utf-8"
columnList = ["inderstryType", 
              "companyCount", "ownerMaleRate", "singlePropCompanyRate", 
              "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
              "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",
              "workerCount", "workerMaleRate", "singlePropWorkerRate", 
              "selfEmpFamilyWorkerRate", "fulltimeWorkerRate", "dayWorkerRate", "etcWorkerRate",
              "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
              "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",
              "avgAge","avgServYear","avgWorkDay","avgTotalWorkTime","avgRegularWorkDay",
              "avgOverWorkDay","avgSalary","avgFixedSalary","avgOvertimeSalary","avgBonusSalary"] 
convertKeyCode = {
    "전체":0,
    "농업임업및어업":1,
    "광업":2,
    "제조업":3,
    "전기가스수도하수":4,
    "건설업":5,
    "도매및소매업":6,
    "운수및창고업":7,
    "숙박및음식점업":8,
    "정보통신업":9,
    "금융및보험업":10,
    "부동산업시설관리지원임대":11,
    "전문과학및기술서비스업":12,
    "교육서비스업":13,
    "보건업및사회복지서비스업":14,
    "오락문화및운동관련서비스업":15,
    "기타공공수리및개인서비스업":16,
}
def getAllDataList():

    dataDir = f"{projectRoot}/{trainInputDir}"
    fileNames = os.listdir(dataDir)

    df = pd.DataFrame(columns=columnList)
    count = 1
    print(len(columnList))
    for fName in fileNames:
        path = f"{dataDir}/{fName}"
        typeName = f"\'{fName.split('.')[0]}\'"
        newDf = pd.read_csv(path, encoding=encoding)
        newList = [typeName,typeName,typeName,typeName,typeName,typeName,typeName,typeName,typeName,typeName]
        newDf['inderstryType'] = newList
        df = pd.concat([df, newDf])

    return df


class ConnectDB:
    def sql_on(self):
        self.lib_Dir = "C:/instantclient-basic-windows.x64-19.27.0.0.0dbru/instantclient_19_27" # instant clinet 받아서 풀어놓고 처리해야 함.
        cx_Oracle.init_oracle_client(lib_dir=self.lib_Dir)

        dsn = cx_Oracle.makedsn("localhost", 1521, sid="xe")
        user = "scott"
        pwd = "tiger"

        try:
            self.connection = cx_Oracle.connect(user=user, password=pwd, dsn=dsn)
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
        dbColumnList = ["id", "year", "inderstryType", 
                    "companyCount", "ownerMaleRate", "singlePropCompanyRate", 
                    "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
                    "U50D100CompanyRate", "U100D300CompanyRate", "U300CompanyRate",
                    "workerCount", "workerMaleRate", "singlePropWorkerRate", 
                    "selfEmpFamilyWorkerRate", "fulltimeWorkerRate", "dayWorkerRate", "etcWorkerRate",
                    "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
                    "U50D100WorkerRate", "U100D300WorkerRate", "U300WorkerRate",
                    "avgAge","avgServYear","avgWorkDay","avgTotalWorkTime","avgRegularWorkDay","avgOverWorkDay","avgSalary","avgFixedSalary","avgOvertimeSalary","avgBonusSalary"] 
        print(len(dbColumnList))
        self.sql_on()
        queryList = []
        queryList.append("""
            CREATE TABLE industryData (
            dataId NUMBER(4) PRIMARY KEY,
            predictYear NUMBER(4),
            inderstryType VARCHAR2(60),
            companyCount NUMBER,
            workerCount NUMBER,
            ownerMaleRate NUMBER(10,4),
            singlePropCompanyRate NUMBER(10,4),
            U1D5CompanyRate NUMBER(10,4),
            U5D10CompanyRate NUMBER(10,4),
            U10D20CompanyRate NUMBER(10,4),
            U20D50CompanyRate NUMBER(10,4),
            U50D100CompanyRate NUMBER(10,4),
            U100D300CompanyRate NUMBER(10,4),
            U300CompanyRate NUMBER(10,4),
            workerMaleRate NUMBER(10,4),
            singlePropWorkerRate NUMBER(10,4),
            selfEmpFamilyWorkerRate NUMBER(10,4),
            fulltimeWorkerRate NUMBER(10,4),
            dayWorkerRate NUMBER(10,4),
            etcWorkerRate NUMBER(10,4),
            U1D5WorkerRate NUMBER(10,4),
            U5D10WorkerRate NUMBER(10,4),
            U10D20WorkerRate NUMBER(10,4),
            U20D50WorkerRate NUMBER(10,4),
            U50D100WorkerRate NUMBER(10,4),
            U100D300WorkerRate NUMBER(10,4),
            U300WorkerRate NUMBER(10,4),
            avgAge NUMBER(10,4),
            avgServYear NUMBER(10,4),
            avgWorkDay NUMBER(10,4),
            avgTotalWorkTime NUMBER(10,4),
            avgRegularWorkDay NUMBER(10,4),
            avgOverWorkDay NUMBER(10,4),
            avgSalary NUMBER(10,4),
            avgFixedSalary NUMBER(10,4),
            avgOvertimeSalary NUMBER(10,4),
            avgBonusSalary NUMBER(10,4)
            )""")

        queryList.append("commit")
        df = getAllDataList()
        yearCount = 2030
        for i in range(len(df.index)):
            if(yearCount >= 2030):
                yearCount -= 10
            data = df.iloc[i]
            queryList.append(f"INSERT INTO industryData VALUES (")
            queryList[len(queryList)-1] += f"{i}, {yearCount}"
            yearCount += 1
            print(df.columns)
            print(len(df.columns))
            for column in range(0, len(df.columns)-1):
                queryList[len(queryList)-1] += f", {data[df.columns[column]]}"
            queryList[len(queryList)-1] += ")"
            

        queryList.append("commit")
        self.sql_execute(queryList)
        self.sql_off()
        pass

db = ConnectDB()
db.ImportDataToOracle()