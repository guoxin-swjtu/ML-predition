from numpy import size
from sklearn import metrics

import pandas as pd
import matplotlib.pylab as plt

plt.rc('font', family='Times New Roman',size=16)

data = pd.read_excel(r'D:\desktop\yhl-rg prediction\test_4(2).xlsx')
data_1 = data['glm-predict1']
data_2 = data['rf-predict2']
data_3 = data['svm-predict3']


z = data['activity']
y_true_1 = z
y_score_1 = data_1

fpr1, tpr1, thresholds = metrics.roc_curve(y_true_1, y_score_1)
roc_auc1 = metrics.auc(fpr1, tpr1)  # the value of roc_auc1
print(roc_auc1)
plt.plot(fpr1, tpr1, 'g', label='GLM = %0.3f' % roc_auc1, linewidth=2)

y_true_2 = z
y_score_2 = data_2

fpr2, tpr2, _ = metrics.roc_curve(y_true_2, y_score_2)
roc_auc2 = metrics.auc(fpr2, tpr2)  # the value of roc_auc2
print(roc_auc2)
plt.plot(fpr2, tpr2, 'navy', label='RF = %0.3f' % roc_auc2, linewidth=2,)

y_true_3 = z
y_score_3 = data_3

fpr3, tpr3, _ = metrics.roc_curve(y_true_3, y_score_3)
roc_auc3 = metrics.auc(fpr3, tpr3)  # the value of roc_auc3
print(roc_auc3)
plt.plot(fpr3, tpr3, 'r', label='SVM = %0.3f' % roc_auc3, linewidth=2,)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

lw=2
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], color='b', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])  # the range of x-axis
plt.ylim([0.0, 1.02])  # the range of y-axis
plt.xlabel('False Positive Rate')  # the name of x-axis
plt.ylabel('True Positive Rate')  # the name of y-axis
#plt.title('Receiver operating characteristic example')  # the title of figure
# plt.grid(linestyle='--')
plt.savefig(r'D:\desktop\yhl-rg prediction\photo\test',bbox_inches='tight',dpi =300)
plt.show()