# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:06:35 2019

@author:luwanrong

E-mail:lwr6608@163.com
"""

import pandas as pd
import numpy as np

data = pd.read_excel('./data/discdata.xls')
'''
183:磁盘容量不变，剔除该数据C:/D:/

184:已使用大小，随时间变化
'''
data['TARGET_ID'].replace(183,np.nan,inplace = True)
data1 = data.dropna(how = 'any')#默认  行  ///如需要列axis = 'colmuns'
data1.index = [i for i in range(len(data1.index))]
result1 = []
result2 = []
for i in data1.index:
    if i % 2 == 0:
        result1.append(data1.iloc[i,5])
    else:
        result2.append(data1.iloc[i,5])
        
result = pd.DataFrame()
result['SYS_NAME'] = data1['SYS_NAME']
result['CWXT_DB_184_C'] = pd.Series(result1)
result['CWXT_DB_184_D'] = pd.Series(result2)
result['COLLECTTIME'] = pd.Series(data1['COLLECTTIME'].unique())
result = result.dropna(how = 'any')

#采用单位根检验（ADF）进行平稳性检验
from statsmodels.tsa.stattools import adfuller as ADF
data = result.iloc[:len(result)-5]
diff = 0
adf = ADF(data['CWXT_DB_184_D'])
print(adf)
while adf[1] >= 0.05:
    diff+=1
    adf = ADF(data['CWXT_DB_184_D'].diff(diff).dropna())
    
print(u'原始序列经过%s阶查分后归于平稳，p值为%s' % (diff,adf[1]))

#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
[[lb],[p]] = acorr_ljungbox(data['CWXT_DB_184_D'],lags=1)
if p < 0.05:
    print(u'原始序列为非白噪声序列，对应的p值为：%s'% p)
else:
    print(u'原始序列为白噪声序列，对应的p值为：%s'% p)
    
[[lb],[p]] = acorr_ljungbox(data['CWXT_DB_184_D'].diff().dropna(),lags=1)

if p < 0.05:
    print(u'一阶查分序列为非白噪声序列，对应的p值为：%s'% p)
else:
    print(u'一阶查分序列白噪声序列，对应的p值为：%s'% p)
result.to_csv('./data/outfile.csv',index = 0)


