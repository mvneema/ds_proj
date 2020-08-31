# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 00:30:34 2020

@author: Neema MV
"""


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

Final_df_salary = pd.read_csv('eda_data.csv')

# choose relevant columns 
Final_df_salary.columns

df_model = Final_df_salary[['job_simp1', 'loc_simp1','desc_len','avg_salary','python_sk', 'spark_sk', 'tableau_sk',
       'aws_sk', 'excel_sk', 'power_sk', 'numpy_sk', 'opencv_sk', 'pandas_sk',
       'nltk_sk', 'machine_sk', 'deep_sk', 'statistics_sk', 'sql_sk','tensorflow_sk','seniority']]

df_model.dtypes

df_model.describe()

df_model.fillna(-99999, inplace=True)

# get dummy data 
df_dum = pd.get_dummies(df_model)

# train test split 
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Multilinear Regression Stats Model & sklearn
from sklearn import linear_model
import statsmodels.api as sm

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

# significant insights
# Hydrebad, Mohali & mumbai tend to pay more here 
# by seniority - pay gets more

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# lasso regression 
lm_l = Lasso(alpha=.99)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

####################### Ensemble Technique #############################
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn import model_selection
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn import metrics

#GradientBoosting
gbmTree = GradientBoostingRegressor(n_estimators=50)
gbmTree.fit(X_train,y_train)
print("gbmTree on training" , gbmTree.score(X_train, y_train))
print("gbmTree on test data ",gbmTree.score(X_test,y_test))
np.mean(cross_val_score(gbmTree,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

#Bagging 
bgcl = BaggingRegressor(n_estimators=100, oob_score= True)
bgcl = bgcl.fit(X_train,y_train)
print("bgcl on train data ", bgcl.score(X_train,y_train))
print("bgcl on test data ", bgcl.score(X_test,y_test))
np.mean(cross_val_score(bgcl,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(gbmTree,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)
gs.best_score_
gs.best_estimator_

# test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_gbm = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_gbm)
mean_absolute_error(y_test,(tpred_lm+tpred_gbm)/2)
## Combination of Linear model & Gradient Boost Model gives least error


