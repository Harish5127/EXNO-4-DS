# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:

STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method


## Developed By: Harish R
## Register no: 212224230085

## CODING AND OUTPUT:
```py
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```
<img width="853" height="259" alt="image" src="https://github.com/user-attachments/assets/8f3b932d-cf12-4210-ad2e-c2cb9b793adf" />


```py
data.isnull().sum()
```
<img width="122" height="303" alt="image" src="https://github.com/user-attachments/assets/40ccdfb9-e950-43b0-890b-5a8bdb569c2b" />


```py
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="821" height="259" alt="image" src="https://github.com/user-attachments/assets/eeedfe90-42a4-432b-bc2c-060da1b6a86b" />

```py
data2=data.dropna(axis=0)
data2
```
<img width="845" height="258" alt="image" src="https://github.com/user-attachments/assets/9e8ce0c8-b362-4d3e-a95b-3abfe87c22bc" />

```py
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="679" height="204" alt="image" src="https://github.com/user-attachments/assets/f18a9240-c3e3-4afe-94d5-0c046203090b" />


```py
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="204" height="265" alt="image" src="https://github.com/user-attachments/assets/f35a5f7f-6196-4cab-a910-c18a60960a43" />

```py
data2
```
<img width="777" height="257" alt="image" src="https://github.com/user-attachments/assets/0f539968-7cc0-4b66-a3c8-bdc1b933b1d8" />

```py
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1012" height="294" alt="image" src="https://github.com/user-attachments/assets/dea19de6-c83b-4818-9ac7-d30539d67a19" />

```py
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="992" height="19" alt="image" src="https://github.com/user-attachments/assets/6790413e-10d8-4e08-a30c-e29ee14f00b8" />

```py
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1001" height="20" alt="image" src="https://github.com/user-attachments/assets/cc9ec314-b47b-4fc5-acf1-022ba6cbed23" />


```py
y=new_data['SalStat'].values
print(y)
```
<img width="101" height="21" alt="image" src="https://github.com/user-attachments/assets/140af1d0-c5dd-4bc5-98ca-81dbd20d153b" />

```py
x=new_data[features].values
print(x)
```
<img width="218" height="81" alt="image" src="https://github.com/user-attachments/assets/07b56291-146f-45b3-afc0-d79a7fa8d639" />


```py

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
<img width="150" height="49" alt="image" src="https://github.com/user-attachments/assets/e0259bb2-87ec-4eef-8735-f3a33a1b2b09" />


```py
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="83" height="37" alt="image" src="https://github.com/user-attachments/assets/cf7395d3-537c-4cc4-92ab-0b03d10fce4c" />

```py
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="105" height="20" alt="image" src="https://github.com/user-attachments/assets/6c7e70e4-7d27-4e91-b67a-145aaa67883e" />

```py
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="147" height="19" alt="image" src="https://github.com/user-attachments/assets/4b6c84bb-8fa6-48ab-a87c-9b90e6e7c4d8" />

```py
data.shape
```
<img width="77" height="26" alt="image" src="https://github.com/user-attachments/assets/0151c6c4-9cb1-4f42-a7ca-759d9201e1c0" />



```py
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="468" height="52" alt="image" src="https://github.com/user-attachments/assets/d6979b0f-ce09-4299-8682-0c1bb7f8d94f" />


```py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="270" height="128" alt="image" src="https://github.com/user-attachments/assets/52b43da9-43ed-4d99-b7fb-168b5266bcd6" />

```py
tips.time.unique()
```
<img width="220" height="35" alt="image" src="https://github.com/user-attachments/assets/f7b3f2a9-b5cc-4f46-908a-a37a1bb1947c" />

```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="112" height="52" alt="image" src="https://github.com/user-attachments/assets/7ec61bcb-1270-4c37-80c1-874adbaf6f0a" />

```py
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="207" height="29" alt="image" src="https://github.com/user-attachments/assets/9731ba3d-eb80-46b0-8b8f-8f48629b1168" />



## RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
