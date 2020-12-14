import pandas as pd
import numpy as np
import sys

# anyway I replace k1 and k2 to k
def replace_to_k(df):
    for _, row in df.iterrows():
        if "k1" == row.at["relation"] or "k2" == row.at["relation"]:
            row.loc["relation"] = "k"
    return df


# 向きを変えたものを出力する
def add_reverses(df):
    def reverse_direction(row):
        copy = row.copy()  # copy for temp
        copy.at["class_a"] = row.at["class_b"]
        copy.at["class_b"] = row.at["class_a"]
        rowtype = row.at["relation"]
        copy.at["relation"] = rowtype.replace("1", "2") if "1" in rowtype else rowtype.replace("2", "1")
        return copy

    for index, row in df.iterrows():
        if "1" in row.at["relation"] or "2" in row.at["relation"]:
            rev = reverse_direction(row)
            df = df.append(rev)
    #print(df)
    df = df.sort_values("title").reset_index(drop=True)
    return df


# 重複削除
def del_confilct(df):
    di = dict()
    for i, row in df.iterrows():
        if row["class_a"] in di.keys() and row["class_b"] == di[row["class_a"]]:
            df.loc[i,"is_conflict"] = True
        else:
            df.loc[i,"is_conflict"] = False
            di[row["class_a"]] = row["class_b"]
            if row["relation"] == "k":
                di[row["class_b"]]=row["class_a"]
    return df

class NonChecker:
    def __init__(self, type, title, source):
        self.type = type
        self.title = title
        self.source = source
        self._di = {}

    # 新しいrelationの追加
    def add_relation(self, row):
        a_name = row["class_a"]
        b_name = row["class_b"]
        try:
            self._di[a_name][b_name] = 1
        except KeyError:
            self._di[a_name] = {}
            self._di[a_name][b_name] = 1
        try:
            self._di[b_name][a_name] = 1
        except KeyError:
            self._di[b_name] = {}
            self._di[b_name][a_name] = 1

    # nonの一覧を返す
    def get_none_rows(self):
        #print(self._di)
        df = None
        stock = list()
        for key1, value1 in self._di.items():# diにはすべてのkeyが含まれている
            stock.append(key1)
            diff = list(set(self._di.keys()) - set(value1.keys())- set(stock)) #差分を取る


            for dkey in diff:
                if df is None:
                    df = self._create_row(key1, dkey)
                else:
                    df = pd.concat([df,self._create_row(key1, dkey)], axis=0)
        return df

    # row を作る
    def _create_row(self, a_name, b_name):

        return pd.DataFrame({"class_a": [a_name], "class_b": [b_name], "relation": ["n"],
                             "type": [self.type], "title": [self.title],
                             "source": [self.source]})


# n (no realation) ラベルのデータを生成、追加する
def add_non_label(df):
    df = df.sort_values('title')  # sort
    nc = None
    for _, row in df.iterrows():
        # 初回ループはNonCheckerを作る
        if nc is None:
            nc = NonChecker(row.at["type"], row.at["title"], row.at["source"])

        # title が変化したら初期化
        if nc.title != row.at["title"]:
            df = pd.concat([df, nc.get_none_rows()], axis=0)  # 結合
            nc = NonChecker(row.at["type"], row.at["title"], row.at["source"])

        # titleが変化しなかったら行を追加
        else:
            nc.add_relation(row)
    none_rows = nc.get_none_rows()
    df = pd.concat([df, none_rows], axis=0)
    return df

def execute(df):
    #df = pd.read_csv("0.dataset_remove_conflict.csv")
    df = replace_to_k(df)
    df = add_reverses(df)
    #df = del_confilct(df)
    #df = add_non_label(df)
    #df.to_csv("1.increased_data_conflict.csv", index=False)
    return df

def main():
    filepath = sys.argv[0]
    df = pd.read_csv(filepath)
    df = pd.execute(df)
    df.to_csv("1.increased_data.csv", index=False)
    #df.to_csv("1.increased_data_conflict.csv", index=False)



if __name__ == "__main__":
    main()