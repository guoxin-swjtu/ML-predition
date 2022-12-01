import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve

plt.rcParams['font.family'] = 'SimHei'  # 配置中文字体
plt.rcParams['font.size'] = 15  # 更改默认字体大小
data = pd.read_excel(r'D:\desktop\yhl-rg prediction\train_data_22.xlsx')#r转义斜杠别忘记
# print(data)
column = data.columns.tolist()[:5]

# 删除某个属性
x = pd.DataFrame(data.drop(['Name'], axis=1))
#
# 归一化会缩小大部分数据之间的差距，这步不是必须，但是个人认为这有利于提高精度
# # 对数据做归一化处理
x_normal = (x - x.min()) / (x.max() - x.min())
# print(x_normal)
# x_normal =np.array(x_normal)
# print(type(x_normal))
#
Y = np.array(x_normal.activity)
X = np.array(pd.DataFrame(x_normal.drop(['activity'], axis=1), index=None))
# print(Y)
# print(X)
# 分割训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# x_train是训练集数据 y_train是训练集的结果
# x_test是测试集数据 y_test是测试集的结果
# test size是测试数据占原始数据规格的0.3
# random_state是划分训练集和测试集的模式，值不变时结果不变，
# print(X_train)
# print(Y_train)

from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 利用数据来训练SVM分类器
clf = SVC(C=0.8,kernel='rbf',gamma=10, class_weight='balanced',probability='Ture')
# kernel核函数 rdf是高斯核 还可以选择其他的线性核等，gamma可调节的影响精度参数
# C是一个惩罚参数，表示对错误分类的惩罚程度，也就是不允许分类出错的程度，默认为1
# class_weight是类别权重，不同类别的权重，如果是balanced则自动根据y值来调整参数
clf.fit(X_train, Y_train)
# y_predict = clf.predict(X_test)

# 计算错误的数目
# error = 0
# for i in range(len(X_test)):
#     if clf.predict([X_test[i]])[0] != Y_test[i]:
#         error += 1
# # 查看错误个数
# print(error)
# # 测试集个数
# print(len(X_test))
# clf = SVC(C=0.8,kernel='rbf',gamma=10, class_weight='balanced',probability='Ture')
y_predict_proba = clf.predict_proba(X_test)
# p_rate = model.get_prob(X)  # 计算分类器把样本分为正例的概率
fpr, tpr, thresh = roc_curve(Y_test, y_predict_proba)
plt.figure(figsize=(5, 5))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot(fpr, tpr)
plt.savefig('roc.png')




# print(clf.score(X_train, Y_train)) # 精度
# y_hat = clf.predict(X_train)
# a = accuracy_score(y_hat, Y_test, '训练集')
# print(clf.score(X_test, Y_test))
# y_hat = clf.predict(X_test)
# show_accuracy(y_hat, Y_test, '测试集')

# from sklearn.neighbors import KNeighborsClassifier as KNN
#
# knc = KNN(n_neighbors=6, )
# knc.fit(X_train, Y_train)
# y_predict = knc.predict(X_test)
# print('KNN准确率', knc.score(X_test, Y_test))
# print('KNN精确率', precision_score(Y_test, y_predict, average='macro'))
# print('KNN召回率', recall_score(Y_test, y_predict, average='macro'))
# print('F1', f1_score(Y_test, y_predict, average='macro'))
# #
# from sklearn.ensemble import RandomForestClassifier
#
# rfc = RandomForestClassifier()
# rfc.fit(X_train, Y_train)
# y_predict = rfc.predict(X_test)
# print('随机森林准确率', rfc.score(X_test, Y_test))
# print('随机森林精确率', precision_score(Y_test, y_predict, average='macro'))
# print('随机森林召回率', recall_score(Y_test, y_predict, average='macro'))
# print('F1', f1_score(Y_test, y_predict, average='macro'))
