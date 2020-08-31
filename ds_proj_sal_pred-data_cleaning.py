# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 00:50:27 2020

@author: Neema MV
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_science=pd.read_csv("datascience.csv")
data_science_specialist=pd.read_csv("datasciencespecialist.csv")
data_analyst=pd.read_csv("dataanalyst.csv")
machine_learning=pd.read_csv("machinelearning.csv")
ai=pd.read_csv("ai.csv")

Final_df = pd.concat([data_science, data_science_specialist, data_analyst, machine_learning, ai], axis = 0).drop_duplicates()
Final_df.shape
Final_df.head(10)
Final_df_sal=Final_df[Final_df['Salary']!= 'None']

def format_salary(row):
    salary = row["Salary"]
    if "-" in salary:
        split = salary.split("-")
        salary_min = split[0]
        salary_max = split[1]
    else:
        salary_min = salary
        salary_max = salary
    
    row["salary_min"] = salary_min.replace("₹","").replace("a month","").replace("a year","").replace(",","")
    row["salary_max"] = salary_max.replace("₹","").replace("a month","").replace("a year","").replace(",","")
       
    if "month" in row["Salary"]:
        row["salary_min"] = int(row["salary_min"])*12
        row["salary_max"] = int(row["salary_max"])*12
    
   
    return row


Final_df_salary = Final_df[Final_df["Salary"]!= "None"].dropna()
Final_df_salary = Final_df_salary.apply(format_salary,axis=1) 

Final_df_salary["salary_min"] = pd.to_numeric(Final_df_salary["salary_min"],'coerce')
Final_df_salary["salary_max"] = pd.to_numeric(Final_df_salary["salary_max"],'coerce')

Final_df_salary["salary_min"].mean()

Final_df_salary['avg_salary'] = (Final_df_salary.salary_min+Final_df_salary.salary_max)/2

import nltk
nltk.download('stopwords')
from nltk import word_tokenize

from nltk.corpus import stopwords
nltk.download('punkt')

def cleanData(desc):
    desc = word_tokenize(desc)
    desc = [word.lower() for word in desc if word.isalpha() and len(word) > 2]
    desc = [word for word in desc if word not in stop_words]
    return desc

stop_words = stopwords.words('english')

tags_Final_df_salary = Final_df_salary["Description"].apply(cleanData)
from collections import Counter
result = tags_Final_df_salary.apply(Counter).sum().items()
result = sorted(result, key=lambda kv: kv[1],reverse=True)
result_series = pd.Series({k: v for k, v in result})

# creating colums for the most used skills
Final_df_salary['python_sk']=Final_df_salary['Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
Final_df_salary.python_sk.value_counts()
#R studio
Final_df_salary['spark_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
Final_df_salary.spark_sk.value_counts()
#Tableau
Final_df_salary['tableau_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
Final_df_salary.tableau_sk.value_counts()
#aws
Final_df_salary['aws_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
Final_df_salary.aws_sk.value_counts()
#excel
Final_df_salary['excel_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
Final_df_salary.excel_sk.value_counts()
#power
Final_df_salary['power_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'power' in x.lower() else 0)
Final_df_salary.power_sk.value_counts()
#numpy
Final_df_salary['numpy_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'numpy' in x.lower() else 0)
Final_df_salary.numpy_sk.value_counts()
#opencv
Final_df_salary['opencv_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'opencv' in x.lower() else 0)
Final_df_salary.opencv_sk.value_counts()
#pandas
Final_df_salary['pandas_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'pandas' in x.lower() else 0)
Final_df_salary.pandas_sk.value_counts()
#nltk
Final_df_salary['nltk_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'nltk' in x.lower() else 0)
Final_df_salary.nltk_sk.value_counts()
#machine 
Final_df_salary['machine_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'machine' in x.lower() else 0)
Final_df_salary.machine_sk.value_counts()
#deep
Final_df_salary['deep_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'deep' in x.lower() else 0)
Final_df_salary.deep_sk.value_counts()
#statistics
Final_df_salary['statistics_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'statistics' in x.lower() else 0)
Final_df_salary.statistics_sk.value_counts()
#sql
Final_df_salary['sql_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
Final_df_salary.sql_sk.value_counts()
#tensorflow
Final_df_salary['tensorflow_sk'] = Final_df_salary['Description'].apply(lambda x: 1 if 'tensorflow' in x.lower() else 0)
Final_df_salary.tensorflow_sk.value_counts()

Final_df_salary.to_csv('salary_data_cleaned.csv',index = False)