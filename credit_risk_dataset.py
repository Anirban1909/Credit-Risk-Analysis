import numpy as np
import pandas as pd
import os 

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


import xgboost as xgb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

os.chdir('D:/Datasets/')
filename="/credit_risk_dataset.csv"
cwd=os.getcwd()
os.getcwd()
cwd 

df=pd.read_csv(cwd+filename)

df.head()
pd.set_option('display.max_columns', 50)


df.isnull().sum()

df['loan_int_rate'].head(50)
df.loc[39:39]
a=df[df['loan_int_rate'].isna()]

a.head()  
a.nunique().count()
b=df[df['person_emp_length'].isna()]
b.head(50)

df['loan_int_rate'].median()
df['person_emp_length'].median()


## replacing the null values with median for loan column
df['loan_int_rate']=df['loan_int_rate'].fillna(df['loan_int_rate'].median())

## replaced the null in emp column with mode for a more conservative approach
df['person_emp_length']=df['person_emp_length'].fillna(0)

## Correlation check

sns.heatmap(df.corr())

### Outliers

sns.set_theme(style="ticks", palette="pastel")


sns.boxplot(df['person_age'], palette=['m','g'])
df=df.drop(df[df['person_age']>70].index)

sns.boxplot(df['person_emp_length'])
df=df.drop(df[df['person_emp_length']>50].index)
df.describe()

sns.pairplot(df)


def create_scatter():
    with plt.style.context('ggplot'):
        fig=plt.figure(figsize=(8,4))
        
        plt.scatter(x=df['person_age'], y=df['person_income'], s=20)
        
        plt.xlabel('Age')
        plt.ylabel('Income')
        
        plt.title("Age vs income plot")
        
create_scatter()     



## Label Encoding a variable

Le=LabelEncoder()
Le1=LabelEncoder()
Le2=LabelEncoder()
Le3=LabelEncoder()

df.head()
df['cb_person_default_on_file']=Le.fit_transform(df['cb_person_default_on_file'])

df['loan_grade']=Le1.fit_transform(df['loan_grade'])
df['person_home_ownership']=Le2.fit_transform(df['person_home_ownership'])
df['loan_intent']=Le3.fit_transform(df['loan_intent'])

### Classes of various encoded variables
df.head() 
Le2.classes_ ### Person Home Ownership
Le.classes_ ###  Person Default
Le3.classes_ ### Loan intent
Le1.classes_ ### Loan Grade

### Visualization of the dataset

avg_home_ownership=df.groupby(by=df['person_home_ownership'])
avg_home_ownership.describe()

person_default_on_file=df.groupby(by=df['cb_person_default_on_file']) 
person_default_on_file

person_default_on_file=pd.DataFrame(person_default_on_file)
person_default_on_file 

g3=df.groupby(['loan_status','person_home_ownership','loan_intent'])

g3.describe()



############################




y=df['loan_status']
X=df.drop(['loan_status','cb_person_cred_hist_length', 'loan_grade'], axis=1)

X.head() 


sns.heatmap(X.corr())

X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.3)


## Creating a test case

test_case=[55,140000,1,7.0,3,55000,12.00,0.6,0] ### 9 variables
test_case=np.array(test_case).reshape(1,-1) ## reshaping it to 2D array

### Random Forest
    
Rf=RandomForestClassifier(n_estimators=100, max_depth=5, verbose=1) 
model1=Rf.fit(X_train,y_train)
y_pred=model1.predict(X_test)

model1.score(X_train, y_train)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
precision_score(y_test, y_pred)

########### XGBOOST    
XGB=xgb.XGBClassifier(eta=0.2, max_depth=4)

model2=XGB.fit(X_train, y_train)
y_pred2=model2.predict(X_test)

model2.score(X_train, y_train)

accuracy_score(y_test, y_pred2)

confusion_matrix(y_test, y_pred2)
precision_score(y_test, y_pred2)

model2.predict(X_test[1011:1012])
X_test[1011:1015]

cvscore1=cross_val_score(XGB, X_train, y_train, cv=5)

cvscore1.mean() 
##### Bagging KNN Classifier

bagging=BaggingClassifier(KNeighborsClassifier(), n_estimators=100)

model3=bagging.fit(X_train, y_train)
y_pred3=model3.predict(X_test)
model3.score(X_train, y_train)

accuracy_score(y_test, y_pred3)

confusion_matrix(y_test, y_pred3)

cvscore=cross_val_score(bagging, X_train, y_train, cv=5)

cvscore 

model3.predict(test_case)

##### ADA boost Classifier

ada=BaggingClassifier(AdaBoostClassifier(), n_estimators=100)

model4=ada.fit(X_train, y_train)
y_pred4=model4.predict(X_test)
model4.score(X_train, y_train)

accuracy_score(y_test, y_pred4)
confusion_matrix(y_test, y_pred4)
precision_score(y_test, y_pred4)

#### Gradient Boosting Classifier

gbc=BaggingClassifier(GradientBoostingClassifier(), n_estimators=150, n_jobs=35, max_samples=200, max_features=7)

model5=gbc.fit(X_train, y_train)
y_pred5=model5.predict(X_test)

model5.score(X_train, y_train)
accuracy_score(y_test, y_pred5)

confusion_matrix(y_test, y_pred5)

model5.predict(test_case)


##### Gaussian Naive Bayes

gnb=BaggingClassifier(GaussianNB(), n_estimators=50)

model6=gnb.fit(X_train, y_train)

y_pred6=model6.predict(X_test)
model6.score(X_train, y_train)
accuracy_score(y_test, y_pred6)
confusion_matrix(y_test, y_pred6)



