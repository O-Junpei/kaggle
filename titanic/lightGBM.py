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