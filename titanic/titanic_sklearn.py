import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 参考
# https://www.codexa.net/kaggle-titanic-beginner/

# データセットをインポート
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# グラフに表示する（今回は使わない）

# 先頭のデータを確認
print(train.head())
print(test.head())

# データフレームのカラム数とデータ数を確認
print(train.shape)
print(test.shape)

# データセットの基本統計量も確認
# データに欠損がないか確認する、漏れがないか確認する
print(train.describe())
print(test.describe())


# DataFrame から欠損地(null) の多い項目を調べる
def missing_value_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    missing_value_table = pd.concat([null_val, percent], axis=1)
    missing_value_table = missing_value_table.rename(columns={0: '欠損数', 1: '%'})
    return missing_value_table


# Age Cabin Embarked でデータが欠損していることを確認
print(missing_value_table(train))
print(missing_value_table(test))

# Age はデータの中央値で埋める
train["Age"] = train["Age"].fillna(train["Age"].median())

# 一番数の多いSで埋める
train["Embarked"] = train["Embarked"].fillna("S")

# Cabin の欠損は今回は使用しないため無視する

# train DF の Cabin 以外のの欠損値がなくなったことを確認する
print(missing_value_table(train))

# test DF の欠損地を埋める
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# test DF の欠損地を埋められたことを確認する
print(missing_value_table(test))

# 文字列を数字に変換する
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 数字に変換できていることを確認
# print(train.head())

# test DF も同様に変換
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 数字に変換できていることを確認
print(test.head())

from sklearn import tree

#
# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# # 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# # 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
#
# # 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)

# print(my_prediction)

# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

# my_tree_one.csvとして書き出し
my_solution.to_csv("sklearn_decision_tree1.csv", index_label=["PassengerId"])

# その２
# 追加となった項目も含めて予測モデルその2で使う値を取り出す
features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 決定木の作成とアーギュメントの設定
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=1)
my_tree_two = my_tree_two.fit(features_two, target)

# tsetから「その2」で使う項目の値を取り出す
test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 「その2」の決定木を使って予測をしてCSVへ書き出す
my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns=["Survived"])
my_solution_tree_two.to_csv("sklearn_decision_tree2.csv", index_label=["PassengerId"])
