# Ex-06  Feature Transformation
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:
### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

## PROGRAM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer 
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()

```

## OUTPUT:
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/3dec2269-138b-44dd-84b6-0173f77ae2bb)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/2483a68c-d6cc-42ea-ba11-7edf3f5bc452)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/ceaec8dc-176c-4463-baa7-8605d50ddc47)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/082134d9-0f40-4fcc-a83f-ad944a1b8db3)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/8d9c194d-a756-461e-85d6-9cd00bcea2a7)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/bdfeb1ba-279e-42b2-bb6f-bc5a102725e8)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/a5d09157-1c90-4a87-b088-59c4997c200b)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/ca4ab138-0e10-4cbd-bf7c-be3509c00dca)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/fa432c68-a8de-48e5-92ce-07d043f909d4)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/5e157327-53e3-497e-a186-57e7cb288a88)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/758ef5fc-eed0-4806-bb00-7493c2079383)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/2fb237d1-3dce-4315-ac39-b051e6e861ff)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/9f1497ed-522f-49ae-9838-07b86ba9f6d0)
![image](https://github.com/NITHISH74/ODD2023-Datascience-Ex06/assets/94164665/2e9d3745-786d-4e71-b0fc-7f08ef0f02e3)



## RESULT:
Thus feature transformation is done for the given dataset.
