# 内田研-研究成果リポジトリ
   この研究を継続してくださる方はakiteruto(attomark)gmail.comに連絡してくれると嬉しいです。
   サポートします！


## ディレクトリ構造
各ファイルは以下のようになっている。
- /dataset :　一番もととなるデータが入ってる、基本手作業で作ってる
    - /create_csv_from_pu : 過去の実験データからデータセットを生成する関連(かなりぐちゃぐちゃ)
        - 中身省略
    - dataset_relation.csv : 一番元となるデータセット
    - wordnet.csv : dataset_relation.csvで出てくる単語に足してwordnet化したもの、重複は削除されているはず
- /exec_backup : 実行時にバックアップとしてcsvを入れたりする用なので基本無視してOK
- /human_exp : 被験者実験用のディレクトリ（結構ごちゃごちゃしてる)
    - /data : 被験者実験の結果が入ってる (ちょっとミスっててデータ重複一部消せてないので注意)
        - 中身省略
    - decoder.ipynb バラバラな実験結果を１つのcsv(ans_df.csv)にまとめるためのスクリプト
    - 0.dataset_remove_conflict.csv : 上のファイルで使った
    - ans_df.csv : 被験者実験データまとめたやつ
    - original.csv : 実験で被験者に渡すファイル(ここらへん一回失敗してるので要確認)
- /scripts : データ前処理用のスクリプトとその依存ファイルが入ってる。 1_preprocessing.ipynbに呼び出される
    - /pretraining_data : 事前学習データ (<span style="color: red; ">gitignoreされているので注意!</span>
        - enwiki_20180420_300d.pkl : wiki2vecの事前学習済みベクトル
        - lexvec.commoncrawl.ngramsubwords.300d.W.pos.vectors : lexvecの事前学習済みベクトル
        - mcs_memo.pickle : mcs_apiのアクセスに時間がかかるため事前にダウンロードしておいたファイル./scripts/create_mcs_csv.pyで生成される
    - features.py : 特徴量を算出してくれるFeatureクラスが入ったスクリプト
    - create_feature.py : features.pyを呼び出して特徴量をdataに追加するためのファイル
    - create_mcs_csv.py : mcs apiにアクセスしてwordnet.csvに入っている単語のis-a群をダウンロードしてくる(create_feature.pyよりも先に実行すること)
    - increase_dataset.py : データセットを受け取って汎化集約を反転させて返す。被験者実験結果も同時に反転処理してる
- 1_preprocessing.ipynb : データの前処理をする。./scripts内のスクリプト呼び出してる。最終的に1_preprocessing_data.csvを出力する
- 2_tuning.ipynb :　optimizerでXGBoostチューニングしてる
- 3_best_model : 交差検証で予測してる。結果考察系の図も出力してる
- 4_feature_analyse.ipynb : 特徴量の分析用 feature_importance算出してる
- 5_figures.ipynb :　learningcurveとか考察用の図を表示してる
- graphs.xml :　論文用のグラフとか表とかExcelでやってる

## 学習済みベクトルのダウンロード先
wiki2vecとlexvecは事前学習済みベクトルをダウンロードして./scripts/pretraining_dataに入れる必要がある。それぞれのダウンロード先は以下の通り。
- wiki2vec : http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2
- lexvec : https://www.dropbox.com/s/mrxn933chn5u37z/lexvec.commoncrawl.ngramsubwords.300d.W.pos.vectors.gz?dl=1

