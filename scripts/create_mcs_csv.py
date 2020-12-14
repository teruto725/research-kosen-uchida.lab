#  wornnetcsvができている前提でmcgのメモを作成する

import pandas as pd
import requests
import pickle
import sys
class MCG():
    @staticmethod
    def get_dict_from_ms_api(word):
        endpoint = "https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance=" + word + "&topK=30"
        result = requests.get(endpoint, verify=False).json()
        return result


#wordnet.csvを受け取ってpretraining_data内に2.mcs_memo.pickleを保存する
def execute(df):
    df = df[~df.duplicated()]
    #create dict
    di = {}
    for _ , row in df.iterrows():
        words = row.loc["word"].split("_")
        for word in words:
            di[word] = MCG.get_dict_from_ms_api(word)

    with open('./scripts/pretraining_data/mcs_memo.pickle', 'wb') as file:
        pickle.dump(di,file)


#第一引数でファイルパスを指定
def main():
    filepath = sys.argv[0]
    df = pd.read_csv(filepath)
    execute(df)

if __name__ == "__main__":
    main()