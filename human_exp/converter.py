import pandas as pd
import requests
import json
import pprint

#0.dataset_relationを被験者用に変換する

data = pd.read_csv("original.csv",index_col= 0)

for i, row in data.iterrows():
    if i % 3 == 0 or i % 8 == 0:
        class_a = row["class_a"]
        class_b = row["class_b"]
        data.loc[i,"class_a"] =  class_b
        data.loc[i,"class_b"] = class_a
data = data.drop_duplicates()
data.to_csv("random.csv",index = None)