import pandas as pd
import numpy as np
import glob
import csv
path = "C:/Users/lhy12/Desktop/PCA/HpA_OA_{}.xlsx.csv"
for m in range(1,10):
   # print(m)
    turnpath  = path.format(m)
    print(turnpath)
   # dataddress = pd.read_csv(turnpath,delimiter=',', usecols=['Time','HA_MISG1','HpA_MISG1'],header= 0)
    col_manual1 = pd.read_csv(turnpath,delimiter=',', usecols=['HA_MISG2'],header= 0)
    col_manual2 = pd.read_csv(turnpath,delimiter=',', usecols=['HA_MISG3'],header= 0)
    col_manual3 = pd.read_csv(turnpath,delimiter=',', usecols=['HpA_MISG2'],header= 0)
    col_manual4 = pd.read_csv(turnpath,delimiter=',', usecols=['HpA_MISG3'],header= 0)
    col_manual5 = pd.read_csv(turnpath,delimiter=',', usecols=['HA_MISG1'],header= 0)
    col_manual6 = pd.read_csv(turnpath,delimiter=',', usecols=['HpA_MISG1'],header= 0)

    col_manual1 = col_manual1.values.tolist()
    col_manual1_array = np.array(col_manual1)
    col_manual2 = col_manual2.values.tolist()
    col_manual2_array = np.array(col_manual2)
    col_manual3 = col_manual3.values.tolist()
    col_manual3_array = np.array(col_manual3)
    col_manual4 = col_manual4.values.tolist()
    col_manual4_array = np.array(col_manual4)
    col_manual5 = col_manual5.values.tolist()
    col_manual5_array = np.array(col_manual5)
    col_manual6 = col_manual6.values.tolist()
    col_manual6_array = np.array(col_manual6)


    print(col_manual1_array[0,0])
    cycle = len(col_manual1)
    print(cycle)
    #print(col_manual_array)
    for col in range(1,m+1):
        HAMISG = 'HA_MISG{}'.format(m)
        HPAMISG = 'HpA_MISG{}'.format(m)
        if m == 1:
            dataddress = pd.read_csv(turnpath, delimiter=',', usecols=['Time', 'HA_MISG1', 'HpA_MISG1'], header=0)
        else:
            dataddress[HAMISG] = 0
            dataddress[HPAMISG] = 0
            for i in range(0, cycle - 5):
                dataddress.loc[i, HAMISG.format(m)] = col_manual5_array[i, 0]
            for i in range(0, cycle - 5):
                dataddress.loc[i, HPAMISG.format(m)] = col_manual6_array[i, 0]
            else:
                print(m)
        for i in range(0,cycle-5):
            dataddress.loc[180+i,HAMISG.format(m)] = col_manual1_array[i,0]
        for i in range(0,cycle-5):
            dataddress.loc[360+i,HAMISG.format(m)] = col_manual2_array[i,0]
        for i in range(0,cycle-5):
            dataddress.loc[180+i,HPAMISG.format(m)] = col_manual3_array[i,0]
        for i in range(0,cycle-5):
            dataddress.loc[360+i,HPAMISG.format(m)] = col_manual4_array[i,0]
    dataddress.to_csv("C:/Users/lhy12/Desktop/PCA/HpA_OA_merge.csv", index=False)
#col_manual.loc[182,'HA_MISG1']
#dataddress.to_csv("C:/Users/lhy12/Desktop/PCA/test.csv")
#??????????????????????????????????????????????????????
#?????????????????????????????????????????????+merge??????????????????
#????????????1????????????1???2???3????????????1?????????5???6????????????4??????
#??????merge
#??????2?????????????????????for??????
#over
#a = pd.re