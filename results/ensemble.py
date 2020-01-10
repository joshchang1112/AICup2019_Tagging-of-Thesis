import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


input_1 = pd.read_csv("7.csv") # 1/11
input_2 = pd.read_csv("new_7.csv") # 1/11
input_3 = pd.read_csv("8.csv") # 1/11
input_4 = pd.read_csv("8-2.csv") # 1/11
input_5 = pd.read_csv("8-3.csv") # 1/11
input_6 = pd.read_csv("8-4.csv") # 2/11 
input_7 = pd.read_csv("8-5.csv") # 0.5/11
input_8 = pd.read_csv("9.csv") # 1/11
input_9 = pd.read_csv("9-2.csv") # 2/11
input_10 = pd.read_csv("8-6.csv")# 0.5/11 


sample = pd.read_csv('../data/task1_sample_submission.csv')

# public predictions
# for i in range(131166):
# private predictions
for i in range(131166, 262948):
    tmp = 0
    max_id = -1
    max_num = -1
    if i % 1000 == 0:
        print(i)
    for j in range(6):
        sample.iloc[i, j+1] = (float(input_1.iloc[i, j+2]) + float(input_2.iloc[i, j+2]) + float(input_3.iloc[i, j+2]) + float(input_4.iloc[i, j+2]) + float(input_5.iloc[i, j+2]) + 2 * float(input_6.iloc[i, j+2]) + 0.5 * float(input_7.iloc[i, j+2]) + float(input_8.iloc[i, j+2]) + 2 * float(input_9.iloc[i, j+2]) + 0.5 * float(input_10.iloc[i, j+2])) / 11
        if sample.iloc[i, j+1]  >= 0.5:
            sample.iloc[i, j+1] = 1
        else:
            sample.iloc[i, j+1] = 0

sample.to_csv('ensemble.csv')
