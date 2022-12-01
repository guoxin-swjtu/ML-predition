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
column = data.columns.tolist()[:4]
#print(column)
# 删除某个属性
x = pd.DataFrame(data.drop(['Name'], axis=1))
#
# print(x)
# 归一化会缩小大部分数据之间的差距，这步不是必须，但是个人认为这有利于提高精度
# # 对数据做归一化处理
x_normal = (x - x.min()) / (x.max() - x.min())
# print(x_normal)
# x_normal =np.array(x_normal)
# print(type(x_normal))
#
Y = np.array(x_normal.activity)#读出是否
X = np.array(pd.DataFrame(x_normal.drop(['activity'], axis=1), index=None))#读出影响因素
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
# print(X_test)

from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 利用数据来训练SVM分类器
clf = SVC(C=0.8,kernel='rbf',gamma=10, class_weight='balanced',probability=True)
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

y1_predict_proba = clf.predict_proba(X_test)

# print(y_predict_proba)
# print(y_predict_proba.shape)
y1_predict_proba_right = y1_predict_proba[:,1]
# print(y_predict_proba_right.shape)
# print(y_predict_proba_right)

# y_predict_proba_right = clf.get_proba(y_predict_proba)  # 计算分类器把样本分为正例的概率
# print(y_predict_proba_right)
fpr1, tpr1, thresh = roc_curve(Y_test, y1_predict_proba_right)
# plt.figure(figsize=(5, 5))
# plt.title('ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.grid(True)
roc_auc1 = metrics.auc(fpr1, tpr1)
plt.plot(fpr1, tpr1,'navy', label='SVM = %0.3f' % roc_auc1, linewidth=2,)

from sklearn.neighbors import KNeighborsClassifier as KNN

knc = KNN(n_neighbors=6, )
knc.fit(X_train, Y_train)
y2_predict_proba = knc.predict_proba(X_test)
y2_predict_proba_right = y2_predict_proba[:,1]
fpr2, tpr2, thresh = roc_curve(Y_test, y2_predict_proba_right)
roc_auc2 = metrics.auc(fpr2, tpr2)
plt.plot(fpr2, tpr2,'r', label='KNN = %0.3f' % roc_auc2, linewidth=2,)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
y3_predict_proba = rfc.predict_proba(X_test)
y3_predict_proba_right = y3_predict_proba[:,1]
fpr3, tpr3, thresh = roc_curve(Y_test, y3_predict_proba_right)
roc_auc3 = metrics.auc(fpr3, tpr3)
plt.plot(fpr3, tpr3,'g', label='RF = %0.3f' % roc_auc3, linewidth=2,)

#plt.savefig(r'D:\desktop\yhl-rg prediction\photo\test',bbox_inches='tight',dpi =300)

# from sklearn.linear_model import LinearRegression
# glm =LinearRegression()
# glm.fit(X_train, Y_train)
# y4_predict_proba = glm.predict_proba(X_test)
# y4_predict_proba_right = y4_predict_proba[:,1]
# fpr4, tpr4, thresh = roc_curve(Y_test, y4_predict_proba_right)
# roc_auc4 = metrics.auc(fpr4, tpr4)
# plt.plot(fpr4, tpr4,'g', label='RF = %0.3f' % roc_auc4, linewidth=2,)

from sklearn.linear_model import LogisticRegression
lr =LogisticRegression()
lr.fit(X_train, Y_train)
y4_predict_proba = lr.predict_proba(X_test)
y4_predict_proba_right = y4_predict_proba[:,1]
fpr4, tpr4, thresh = roc_curve(Y_test, y4_predict_proba_right)
roc_auc4 = metrics.auc(fpr4, tpr4)
plt.plot(fpr4, tpr4,'m', label='LR = %0.3f' % roc_auc4, linewidth=2,)


lw=2
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], color='b', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])  # the range of x-axis
plt.ylim([0.0, 1.02])  # the range of y-axis
plt.xlabel('False Positive Rate')  # the name of x-axis
plt.ylabel('True Positive Rate')  # the name of y-axis
plt.grid(linestyle='-.')
plt.savefig(r'D:\desktop\yhl-rg prediction\photo\test-pp',bbox_inches='tight',dpi =300)
plt.show()