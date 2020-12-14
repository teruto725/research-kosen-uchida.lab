"""
incleaseddataから特徴量を作成するファイル。
前提として
- wordnet.csv
- mcs_data.pickle
- enwiki.pkl
- lexvec.vectors
が必要

実行時間かかるから注意

"""



import numpy as np
import gensim
import requests
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import pandas as pd
from wikipedia2vec import Wikipedia2Vec
import pickle
from scripts._features import LexVec,Wiki2vec,MCG,WordNet,Feature,BERT
import sys

#setup系
#lexvec
def setup(wordnet_df):
    print("Start")
    PATH = './scripts/pretraining_data/lexvec.commoncrawl.ngramsubwords.300d.W.pos.vectors'
    lexw2v = gensim.models.KeyedVectors.load_word2vec_format(PATH)
    print("Lexvec is completed")
    #siki2vwec
    wiki2vec = Wikipedia2Vec.load("./scripts/pretraining_data/enwiki_20180420_300d.pkl")
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    print("Wiki2vec is completed")
    #wordnet
    print("Wordnet is completed")
    #mcg
    mgc_di = None
    with open('./scripts/pretraining_data/mcs_memo.pickle', 'rb') as file:
        mgc_di= pickle.load(file)
    print("Mcg is completed")

    LexVec.setup(lexw2v)
    Wiki2vec.setup(wiki2vec)
    MCG.setup(mgc_di)
    WordNet.setup(brown_ic,wordnet_df)
    BERT.setup()

    print("Setup sequence is all green")

    print("Proceed to creating feature sequence")
    # df作成


#wordnet.csvを受け取ってpretraining_data内に2.mcs_memo.pickleを保存する
def execute(df,wordnet_df):
    setup(wordnet_df)
    feature_names = ["sim_by_mcg",
                    "is_a_by_mcg",
                    "lexvec_sim",
                    "wiki2vec_sim",
                    "wn_hu_average",
                    "wn_pu_average",
                    "wn_hu_best",
                    "wn_pu_best",
                    "wn_sim_path",
                    "wn_sim_lch",
                    "wn_sim_wup",
                    "wn_sim_res",
                    "wn_sim_jcn",
                    "wn_sim_lin",
                    "has_same_word",
                    "is_include_word",
                    "bert_sim",
                    "lexvec_diff",
                    "wiki2vec_diff",
                    "bert_diff"
                    ]
    for index, row in df.iterrows():
        #print(index)
        for name in feature_names:
            df.loc[index,name] = Feature.evaluate(row.loc["class_a"],row.loc["class_b"],name)
    print("All feature is completed")
    return df

#第一引数でファイルパスを指定
def main():
    increased_data_path = sys.argv[0]
    wordnet_data_path = sys.argv[1]
    df = pd.read_csv(increased_data_path)
    wordnet_df = pd.read_csv(wordnet_data_path)
    execute(df,wordnet_df)
    df.to_csv("3.add_features_data.csv",index = None)

if __name__ == "__main__":
    main()