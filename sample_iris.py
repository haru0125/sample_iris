import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pld
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

# irisデータセットをロード
iris_dataset = load_iris()

# description
print(iris_dataset['DESCR'])

# keyセット確認
print(iris_dataset.keys())

# 分類のターゲット（正解）
print(iris_dataset['target_names'])

# feature_names: 特微量（ガクの長さ, ガクの幅, 花弁の長さ, 花弁の幅）
print(iris_dataset['feature_names'])

# 正解値
#print(iris_dataset['target'])

# DataFrame生成
# data = pld.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
# print(data)

# 訓練データとテストデータに分割する
# X: 学習対象データ、Y: 正解値
# _train: 学習データ、_test: 学習後に当てるテストデータ
# test_size: テストデータに分割するパーセンテージ
# random_state: 振り分けに利用するrandomのseed
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.3, random_state=0)
#data = pld.DataFrame(X_train, columns=iris_dataset.feature_names)
#print(data)
#print(y_train)


# 訓練データからDataFrame作成
iris_dataframe = pld.DataFrame(X_train, columns=iris_dataset.feature_names)

# 訓練データのDataFrameからscatter_matrixを作成
# c: 指定した値で色を変更する
# figsize: pivotのサイズ
# marker: pivotの文字
# alpha: 透明度。1だと重なりが見づらい
# cmap=mglearn.cm3 color-map
grr = scatter_matrix(frame=iris_dataframe, c=y_train, figsize=(15, 15), marker='x', hist_kwds={'bins': 20}, alpha=.8, cmap=mglearn.cm3)
#plt.show() # 表示
plt.savefig("iris.png") # ファイル出力

# 学習 - K近傍法
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

# 訓練データからモデルを構築
# knnを出力すると設定パラメータが参照可能
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')

# 新しいデータを設定（ガクの長さ, ガクの幅, 花弁の長さ, 花弁の幅）
X_new1 = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction1 = knn.predict(X_new1)
print(iris_dataset['target_names'][prediction1])

# テストデータをモデルに投入
y_pred = knn.predict(X_test)

# 予測結果とテストデータの合致率から精度を算出
# y_pred: 予測結果
# y_test: 正解
print(np.mean(y_pred == y_test))
