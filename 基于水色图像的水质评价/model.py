# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:57:37 2019

@author:luwanrong

E-mail:lwr6608@163.com
"""

#导入数据
import pandas as pd

inputfile = './data/modelfile.csv'
data = pd.read_csv(inputfile)
data = data.as_matrix()

#划分
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data[:,0:9]*30,data[:,9],test_size=0.2, random_state=42)
#*30用来放大特征

#导入模型相关的函数，建立并且训练模型
from sklearn import svm

model = svm.SVC()
model.fit(x_train, y_train)

#导入输出相关的库，生成混淆矩阵
from sklearn import metrics

cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))
print(cm_train)
print(cm_test)

#报告
print(metrics.classification_report(y_train, model.predict(x_train)))
print(metrics.classification_report(y_test, model.predict(x_test)))

#准确率
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, model.predict(x_train)))
print(accuracy_score(y_test, model.predict(x_test)))