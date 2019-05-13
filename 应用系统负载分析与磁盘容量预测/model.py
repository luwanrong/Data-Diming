# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:00:27 2019

@author:luwanrong

E-mail:lwr6608@163.com
"""

import pandas as pd

data = pd.read_csv('./data/outfile.csv',index_col = 'COLLECTTIME')#将collecttime设置为索引
data = data.iloc[:len(data)-5]
xdata = data['CWXT_DB_184_D']
from statsmodels.tsa.arima_model import ARIMA
'''
pmax = int(len(xdata)/10)
qmax = int(len(xdata)/10)

bic_matrix = []
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try:
            tmp.append(ARIMA(xdata,(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
    
bic_matrix = pd.DataFrame(bic_matrix)
#处理空值nan
bic_matrix.fillna(float('inf'),inplace = True)

p,q = bic_matrix.stack().idxmin()
print(u'BIC最小的p值和q值为：%s，%s'%(p,q))
'''
#模型检验

arima = ARIMA(xdata,(1,1,1)).fit()
#预测
xdata_pred = arima.predict(typ = 'levels')
pres_error = (xdata_pred - xdata).dropna()
from statsmodels.stats.diagnostic import acorr_ljungbox
lb,p = acorr_ljungbox(pres_error,lags=12)
h = (p<0.05).sum()
if h>0:
    print(u'模型ARIMA(1,1,1)不符合白噪声检验')
    
else:
    print(u'模型ARIMA(1,1,1)符合白噪声检验')
#模型预测
#模型预测
test_predict=arima.forecast(5)#预测未来5个值
print (test_predict)
#预测对比
test_data=pd.DataFrame(columns=[u'实际容量',u'预测容量'])
test_data[u'实际容量']=data[(len(data)-5):]['CWXT_DB_184_D']
test_data[u'预测容量']=test_predict[0]
#test_data = test_data.applymap(lambda x: '%.2f' % x)


#模型评价
abs_ = (test_data[u'预测容量']-test_data[u'实际容量']).abs()
mae_ = abs_.mean() # mae平均绝对误差  
rmas_ = ((abs_**2).mean())**0.5 #rmas均方根误差  
mape_ = (abs_/test_data[u'实际容量']).mean()/10**6 #mape平均绝对百分误差  
print(abs_ ) 
print(mae_ ) 
print(rmas_) 
print(mape_ ) 
errors = 1.5  
print('误差阈值为%s' % errors)  
if (mae_ < errors) & (rmas_ < errors) & (mape_ < errors):  
    print(u'平均绝对误差为：%.4f, \n均方根误差为：%.4f, \n平均绝对百分误差为：%.4f' % (mae_, rmas_, mape_))  
    print('误差检验通过！') 
else:  
    print('误差检验不通过')
