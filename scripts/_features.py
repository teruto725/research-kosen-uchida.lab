import numpy as np
from nltk.corpus import wordnet as wn
import torch
import transformers as ppb # pytorch transformers
# lexvec
class LexVec():

    @staticmethod
    def setup(w2v):
        LexVec.w2v = w2v

    @staticmethod
    def word_sim(word_a, word_b):
        return LexVec.w2v.similarity(word_a, word_b)

    @staticmethod
    def get_vec(class_a):
        vec = None  # ndarray
        con = 0
        li_a = class_a.split("_")
        for a in li_a:
            if vec is None:
                try:
                    vec = LexVec.w2v[a]
                    con += 1
                except KeyError:
                    return None
            else:
                try:
                    vec = vec + LexVec.w2v[a]
                    con += 1
                except KeyError:
                    return None
        return vec / con

# wiki2vec
class Wiki2vec():
    @staticmethod
    def setup(wiki2vec):
        Wiki2vec.wiki2vec = wiki2vec

    @staticmethod
    def word_sim(word_a, word_b):
        try:
            w2v_a = Wiki2vec.wiki2vec.get_word_vector(word_a)
            w2v_b = Wiki2vec.wiki2vec.get_word_vector(word_b)
            return np.dot(w2v_a, w2v_b) / (np.linalg.norm(w2v_a) * np.linalg.norm(w2v_b))
        except:  # 登録されていない単語
            return np.nan

    #wiki2vecで複合語の単語ベクトルを返す
    @staticmethod
    def get_vec(class_a):
        vec = None #ndarray
        con = 0
        li_a = class_a.split("_")
        for a in li_a:
            if vec is None:
                try:
                    vec = Wiki2vec.wiki2vec.get_word_vector(a)
                    con += 1
                except KeyError:
                    return None
            else:
                try:
                    vec = vec + Wiki2vec.wiki2vec.get_word_vector(a)
                    con += 1
                except KeyError:
                    return None
        return vec/con



# MicrosoftConceptGraphA
# API取得に時間がかかるため予め2.create_mcs_csv.pyを回すことでダウンロードしておく
class MCG():
    @staticmethod
    #予めダウンロードしておいた辞書データをセットする
    def setup(di):
        MCG.di = di  #

    @staticmethod
    def _get_dict_from_ms_api(word):
        if word in MCG.di.keys():
            return MCG.di[word]
        else:
            print("ERROR in MCG:" + word + "is none")

    @staticmethod
    def eva_is_a(word_a, word_b):
        di_a = MCG._get_dict_from_ms_api(word_a)#word_aのis-aワード上位30個の辞書{"単語":評価値}を取得する
        if di_a == {}:
            return np.nan#word_aがMCG上に登録されていなければ欠損値とする
        if word_b in di_a.keys():
            return di_a[word_b] * 100#word_bがdi_aに存在していれば評価値*100の値を返す
        else:
            return 0# 存在しなければ0を返す

    @staticmethod
    def word_sim(word_a, word_b):
        di_a = MCG._get_dict_from_ms_api(word_a)
        di_b = MCG._get_dict_from_ms_api(word_b)
        eva = 0
        if di_a == {} or di_b == {}:
            return np.nan
        for key, value in di_a.items():
            if key in di_b.keys():
                eva += 10 * (value + di_b[key])
        return np.average(eva)


# wordnet
class WordNet():
    memo = {}

    @staticmethod
    def setup(brown_ic, wn_df):
        WordNet.brown_ic = brown_ic
        WordNet.wn_df = wn_df

    # strを受け取ってsynsetオブジェクトに変換して返す
    @staticmethod
    def get_wn(word):
    
        word_wn = (
        WordNet.wn_df[WordNet.wn_df["word"] == word].loc[WordNet.wn_df[WordNet.wn_df["word"] == word].index[0], "wn"])
        # print(word_wn)g
        return wn.synset(word_wn)

    @staticmethod
    def sim_path(word_a, word_b):
        wn_a = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        return wn_a.path_similarity(wn_b)

    @staticmethod
    def sim_lch(word_a, word_b):
        wn_a = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        return wn_a.path_similarity(wn_b)

    @staticmethod
    def sim_wup(word_a, word_b):
        wn_a = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        return wn_a.wup_similarity(wn_b)

    @staticmethod
    def sim_res(word_a, word_b):
        wn_a = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        return wn_a.res_similarity(wn_b, WordNet.brown_ic)

    @staticmethod
    def sim_jcn(word_a, word_b):
        wn_a = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        return wn_a.jcn_similarity(wn_b, WordNet.brown_ic)

    @staticmethod
    def sim_lin(word_a, word_b):
        wn_a = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        return wn_a.lin_similarity(wn_b, WordNet.brown_ic)

    @staticmethod
    def hu(word_a, word_b):
        wn_a = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        if wn_a == wn_b:
            return 1.0
        coms = wn_a.lowest_common_hypernyms(wn_b)  # listで出てくるので注意list[0]が一番近いてか理想は自分ずれてる時だけずれる
        stack = []
        for com in coms:
            if wn_a == com:
                return 1.0
            stack.append(wn_a.path_similarity(com))
        return max(stack)

    @staticmethod
    def pu(word_a, word_b):
        part_wn = WordNet.get_wn(word_a)
        wn_b = WordNet.get_wn(word_b)
        max_pu = 0.0
        if part_wn.part_holonyms() != []:  # 集約かつ子クラスが部分語だったら
            part_holo_wns = part_wn.part_holonyms()  # 全体語に置き換える（ここ書き換えること）
            for part_holo_wn in part_holo_wns:
                pu = part_holo_wn.path_similarity(wn_b)
                max_pu = max(max_pu, pu)
        return max_pu


# BERT (wordnetに依存しているのでwordnetのsetupを終わらしてからsetupすること）
class BERT():
    @staticmethod
    def setup():
        # bertのimport
        model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        bert_df = WordNet.wn_df.loc[:, ["word"]]
        for i, row in bert_df.iterrows():
            bert_df.loc[i, "define"] = WordNet.get_wn(row.loc["word"]).definition()
        # bertのトークン化
        bert_df.loc[:, "token"] = bert_df.loc[:, "define"].apply(
            (lambda x: tokenizer.encode(x, add_special_tokens=True)))
        # arrayに変換
        max_len = 0
        for i in bert_df["token"].values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in bert_df["token"].values])
        input_ids = torch.tensor(np.array(padded))
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        vec_arr = last_hidden_states[0][:, 0, :].numpy()  # vector array
        word_arr = bert_df.loc[:, "word"].values  # word name array
        BERT.vec_di = {}
        for i in range(len(word_arr)):
            BERT.vec_di[word_arr[i]] = vec_arr[i]

    @staticmethod
    def vectorize(class_a):#ベクトル化する
        vec = None  # ndarray
        li_a = class_a.split("_")
        for word in li_a:
            if vec is None:
                vec = BERT.vec_di[word]
                continue
            vec += BERT.vec_di[word]
        vec = vec / len(li_a)
        return vec

    @staticmethod
    def vec_diff(class_a, class_b):#ベクトルの差分を出力する
        vec_a = BERT.vectorize(class_a)
        vec_b = BERT.vectorize(class_b)
        return (vec_a - vec_b)

    @staticmethod
    def cos_sim(class_a,class_b):
        vec_a = BERT.vectorize(class_a)
        vec_b = BERT.vectorize(class_b)
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))



class Common():
    @staticmethod
    def equal(word_a,word_b):
        return word_a == word_b


class Feature():
    # class_a スペース区切りの語句
    @staticmethod
    def evaluate(class_a, class_b, name):
        if name == "sim_by_mcg":
            return Feature._average(class_a, class_b, MCG.word_sim)
        if name == "is_a_by_mcg":
            return Feature._average(class_a, class_b, MCG.eva_is_a)
        if name == "lexvec_sim":
            return Feature._average(class_a, class_b, LexVec.word_sim)
        if name == "lexvec_diff":
            return Feature._vec_diff(class_a,class_b,LexVec.get_vec)
        if name == "wiki2vec_sim":
            return Feature._average(class_a, class_b, Wiki2vec.word_sim)
        if name == "wiki2vec_diff":
            return Feature._vec_diff(class_a,class_b,Wiki2vec.get_vec)
        if name == "wn_hu_average":
            return Feature._average(class_a, class_b, WordNet.hu)
        if name == "wn_pu_average":
            return Feature._average(class_a, class_b, WordNet.pu)
        if name == "wn_hu_best":
            return Feature._get_best(class_a, class_b, WordNet.hu)
        if name == "wn_pu_best":
            return Feature._get_best(class_a, class_b, WordNet.pu)
        if name == "wn_sim_path":
            return Feature._average(class_a, class_b, WordNet.sim_path)
        if name == "wn_sim_lch":
            return Feature._average(class_a, class_b, WordNet.sim_lch)
        if name == "wn_sim_wup":
            return Feature._average(class_a, class_b, WordNet.sim_wup)
        if name == "wn_sim_res":
            return Feature._average(class_a, class_b, WordNet.sim_res)
        if name == "wn_sim_jcn":
            return Feature._average(class_a, class_b, WordNet.sim_jcn)
        if name == "wn_sim_lin":
            return Feature._average(class_a, class_b, WordNet.sim_lin)
        if name == "bert_sim":
            return BERT.cos_sim(class_a, class_b)
        if name == "bert_diff":
            return Feature._vec_diff(class_a,class_b,BERT.vectorize)
        if name == "has_same_word":
            return Feature._sum(class_a, class_b, Common.equal)
        if name == "is_include_word":
            return Feature._include(class_a, class_b)

    @staticmethod
    def _get_best(class_a, class_b, func):
        li_a = class_a.split("_")
        li_b = class_b.split("_")
        stack = []
        for a in li_a:
            for b in li_b:
                stack.append(func(a, b))
        return max(stack)

    @staticmethod
    def _average(class_a, class_b, func):
        li_a = class_a.split("_")
        li_b = class_b.split("_")
        stack = []
        for a in li_a:
            for b in li_b:
                stack.append(func(a, b))
        return np.average(stack)

    @staticmethod
    def _sum(class_a, class_b, func):
        li_a = class_a.split("_")
        li_b = class_b.split("_")
        stack = []
        for a in li_a:
            for b in li_b:
                stack.append(func(a, b))
        return sum(stack)

    @staticmethod
    def _include(class_a, class_b):
        li_a = class_a.split("_")
        li_b = class_b.split("_")
        if len(li_a) == 1:
            if li_a[0] in li_b:
                return 1
        return 0

    # ベクトルの差分
    @staticmethod
    def _vec_diff(class_a,class_b,func):
        vec_a = func(class_a)
        vec_b = func(class_b)
        if vec_a is None or vec_b is None:
            return np.nan
        return str(list(vec_a-vec_b))



