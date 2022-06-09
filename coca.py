# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:48:37 2022

@author: Sai pranay
"""

#-------------------------importing_the_dataset--------------------------------

import pandas as pd

ccs = pd.read_excel("E:\\DATA_SCIENCE_ASS\\FORECASTING\\CocaCola_Sales_Rawdata.xlsx")
print(ccs)
list(ccs)
ccs.head()
ccs.shape
ccs.info()
ccs.describe()
ccs.value_counts()
ccs.plot()
ccs.hist()

dates=pd.date_range(start='1986',periods=42,freq='Q')
dates
dd=pd.DataFrame(dates)
dd
ccs['dates']=dates
ccs
#droping
ccs.drop(['Quarter'],axis=1,inplace=True)
ccs
#plot
ccs.Sales.plot()
ccs.dates.plot()
 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

t=[]
for i in range(1,43):
    t.append(i)
t
ccs['t'] = pd.DataFrame(t)    
ccs['log_sales'] = np.log(ccs['Sales'])
ccs['t_sq'] = ccs['t']*ccs['t']
ccs
ccs["Date"] = pd.to_datetime(ccs.dates,format="%b-%y")
ccs["month"] =ccs.Date.dt.strftime("%b") 
ccs["year"] =ccs.Date.dt.strftime("%Y") 
ccs2 = ccs.copy()
ccs2 = pd.get_dummies(ccs, columns = ['month'])
print(ccs2)
list(ccs2)
ccs2.shape
list(ccs2)
list(ccs)
ccs2.shape


#plotting

plt.figure(figsize=(8,6))
sns.boxplot(x="dates",y="Sales",data=ccs)

plt.figure(figsize=(10,8))
heatmap_y_month = pd.pivot_table(data=ccs,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") 

# Splitting data

Train = ccs2.head(32)
Test = ccs2.tail(10)
Train
Test

import statsmodels.formula.api as smf 


#Linear Model
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


#Exponential
Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


#Quadratic 
Quad = smf.ols('Sales~t+t_sq',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sq"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


#Additive seasonality 
add_sea = smf.ols('Sales~month_Mar+month_Jun+month_Sep+month_Dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['month_Mar','month_Jun','month_Sep','month_Dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Sales~t+t_sq+month_Mar+month_Jun+month_Sep+month_Dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['month_Mar','month_Jun','month_Sep','month_Dec','t','t_sq']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


##Multiplicative Seasonality
Mul_sea = smf.ols('log_sales~month_Mar+month_Jun+month_Sep+month_Dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_sales~t+month_Mar+month_Jun+month_Sep+month_Dec',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


#Compare the results 
coladata = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(coladata)

table_rmse=pd.DataFrame(coladata)
table_rmse.sort_values(['RMSE_Values'])


#inference: after fitting the model rmse_add_sea_quad is better and 4 dummies created by month'''