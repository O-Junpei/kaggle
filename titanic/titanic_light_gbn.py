import pandas as pd
import numpy as np

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


# LightGBM
import lightgbm as lgb

# その1
# 下記のパラメータでモデルを学習する
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# モデル作成
model = lgb.LGBMClassifier()
model.fit(features_one, target)

# 予測
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
my_prediction = model.predict(test_features)

# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

# my_tree_one.csvとして書き出し
my_solution.to_csv("light_gbn1.csv", index_label=["PassengerId"])


# その2
# 下記のパラメータでモデルを学習する
target2 = train["Survived"].values
features_one2 = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
#
# モデル作成
model2 = lgb.LGBMClassifier()
model2.fit(features_one2, target2)
#
# 予測
test_features2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction2 = model2.predict(test_features2)

# PassengerIdを取得
PassengerId2 = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution2 = pd.DataFrame(my_prediction2, PassengerId2, columns=["Survived"])

# my_tree_one.csvとして書き出し
my_solution2.to_csv("light_gbn2.csv", index_label=["PassengerId"])