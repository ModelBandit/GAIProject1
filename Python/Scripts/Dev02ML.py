import pandas as pd
import joblib
import copy

model_dir = "./model"
column_list = ["yearData",
              "companyCount", 
              "ownerMaleRate", #"ownerFemaleRate",
            #   "singlePropCompanyRate", #"multiBusinessCompanyRate",
              "U1D5CompanyRate", "U5D10CompanyRate", "U10D20CompanyRate", "U20D50CompanyRate", 
              "U50D100CompanyRate", #"U100D300CompanyRate",# "U300CompanyRate",
              "workerCount", 
              "workerMaleRate",# "workerFemaleRate",
            #   "singlePropWorkerRate",# "multiBusinessWorkerRate",
            #   "selfEmpFamilyWorkerRate", 
              "fulltimeWorkerRate", "dayWorkerRate",# "etcWorkerRate",
              "U1D5WorkerRate", "U5D10WorkerRate", "U10D20WorkerRate", "U20D50WorkerRate", 
              "U50D100WorkerRate",# "U100D300WorkerRate",# "U300WorkerRate",
            #   "avgAge",
            #   "avgServYear","avgWorkDay",
              #"avgTotalWorkTime",
            #   "avgRegularWorkDay",
              "avgOverWorkDay",
              "avgSalary",
            #   "avgFixedSalary","avgOvertimeSalary","avgBonusSalary"
            ]

def run(last_data, target_industry, predict_range):
    df = pd.read_csv(last_data, encoding="utf-8-sig")
    df = df[df["industryType"] == target_industry]
    df = df[column_list]
    
    dfList = []
    newDf = copy.copy(df)
    newDf.values[0] = 2020
    lr = joblib.load(f"{model_dir}/{target_industry}.pkl")

    for year in range(2021,2021+predict_range+1):
        newDf = lr.predict(newDf)
        dfList.append(newDf)
        newDf = copy.copy(newDf)
        newDf[0][0] = year


    for d in dfList:
        print(d)

        
    # run(r"resources\dev02\비율2회분할\target\2019.csv", "광업", 5)