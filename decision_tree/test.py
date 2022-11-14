import numpy as np 
import pandas as pd 
from random import randrange

data_1 = pd.read_csv('train.csv')
examples = data_1#.iloc[:, :-1]
#print(examples.iloc[:,-1:])
#mode = examples.iloc[:,-1:].mode()
df = pd.DataFrame({'c1': [1, 2, 3, 5, 2], 'c2': [10, 11, 10, 11, 11]})
mode = df.iloc[: , -1:].mode()

for v in df.iloc[:, 1].unique(): #10,11
    exs = df.drop(df[df.iloc[:,1] != v].index)
    print(exs)
    print("--")
    for u in exs.iloc[:, 0].unique(): 
        foo = exs.drop(exs[exs.iloc[:,0] != u].index)
        print(foo)
        print()



# print(len(mode.index))
# if len(mode.index) == 1:
#     print(mode.iloc[0][0])
# else: 
#     print(mode.iloc[randrange(len(mode.index))][0])
#
# print(len(df.iloc[: , -1].unique()))
# print(df.iloc[: , -1][0])

# print(df.iloc[: , -1].unique())

# for v in df.iloc[:,1].unique():
#     exs = df.drop(df[df.iloc[:,1] != v].index)
#     print(exs)

# list1 = [1, 2, 3, 5]
# list2 = list1[:2]
# a = 0
# print(list1[:a] + list1[a+1:])
