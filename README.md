# Data Science Salary Estimator: Project Overview

* Created a tool that estimates data science salaries (MAE ~ $ 11K) to help data scientists negotiate their income when they get a job.
* Scraped over 1000 job descriptions from glassdoor using python and selenium
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark.
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.

# Code and Resources Used
Python Version: 3.7
Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, beautifulsoup
For Web Framework Requirements: pip install -r requirements.txt
Scraper Github: https://github.com/codeheroku/Introduction-to-Machine-Learning/tree/master/Indeed%20Job%20Analysis

# YouTube Project Walk-Through
https://www.youtube.com/watch?v=QiD1lbM-utk&t=600s

# Web Scraping
Tweaked the web scraper github repo (above) to scrape 1000 job postings from glassdoor.com. With each job, we got the following:

Title
Location
Company 
Salary
Description

# Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

* Parsed numeric data out of salary
* Made columns for employer provided salary minimum and maximum
* Removed rows without salary
* Made columns for if different skills were listed in the job description:
  * Python
  * R
  * Excel
  * AWS
  * Spark
  * SQL
  * Tableau
  * Power BI
  * Numpy
  * Open CV
  * Pandas
  * NLTK
  * Machine Learning
  * Deep Learning
  * Statistics
  * Tenserflow
  * Predictive
* Column for simplified job title and Seniority
* Column for description length

# EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables.

![](/EDA1.PNG)

![](/EDA3.PNG)

![](/EDA2.PNG)

# Model Building
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.
 
I tried several different models:

Multiple Linear Regression – Baseline for the model
Lasso Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.
Gradient Boosting Regressor & Bagging to see how the model fits the data.

# Model performance
The Combination of Linear model & Gradient Boosting model far outperformed the other approaches on the test and validation sets.

* Gradient Boosting Regressor : MAE = 337453.71
* Linear Regression: MAE = 440621.25
* Ridge Regression: MAE = 445934.85
