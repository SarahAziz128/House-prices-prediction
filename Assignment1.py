from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

%matplotlib inline

data= pd.read_csv("../input/test.csv")
data.head()
data.describe(include=[np.number])
data.isnull().sum()  #Data not having any NaNs

names=['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','Utilities','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','OverallOual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']
df=data[names]
correlations= df.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=np.arange(0,15,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

data['MSZoning'] = data['MSZoning'].astype('category',ordered=True)
data['Street'] = data['Street'].astype('category',ordered=True)
data['Alley'] = data['Alley'].astype('category',ordered=True)
data['LotShape'] = data['LotShape'].astype('category',ordered=False)
data['Utilities'] = data['Utilities'].astype('category',ordered=False)
data['Condition1'] = data['Condition1'].astype('category',ordered=False)
data['Condition2'] = data['Condition2'].astype('category',ordered=False)
data['RoofMatl'] = data['RoofMatl'].astype('category',ordered=False)
data['Exterior1st'] = data['Exterior1st'].astype('category',ordered=True)
data['Exterior2nd'] = data['Exterior2nd'].astype('category',ordered=True)
data['SaleCondition'] = data['SaleCondition'].astype('category',ordered=True)



data.dtypes
#sns.set_style()
sns.regplot(x='MSSubClass',y='Saleprice',data=data)
sns.regplot(x='LotFrontage',y='Saleprice',data=data)
sns.regplot(x='YearBuilt',y='Saleprice',data=data)
sns.stripplot(x='YearRemodAdd', y='Saleprice',data=data)
sns.stripplot(x='MasVnrArea', y='Saleprice',data=data, size=5)
sns.stripplot(x='BsmtFinF2', y='Saleprice',data=data, size=5)
sns.stripplot(x='BsmtUnfSF', y='Saleprice',data=data, size=5)
sns.stripplot(x='TotalBsmtSF', y='Saleprice',data=data, size=5)
sns.stripplot(x='1stFlrSF', y='Saleprice',data=data, size=5)
sns.stripplot(x='2ndFlrSF', y='Saleprice',data=data, size=5)
data=data[data['OverallOual'] < 10]
data=data[data['OverallCond']<10]
data.head()


c=['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','SaleType','SaleCondition','OverallOual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation']
df=data[c]
df=pd.get_dummies(df,columns=['grade'], drop_first=True)
y=data['Saleprice']

x_train,x_test,y_train,y_test=train_test_split(df,y,train_size=0.8,random_state=42)

x_train.head()
reg=LinearRegression()
reg.fit(x_train,y_train)
print('Coefficients: \n', reg.coef_)
print(metrics.mean_squared_error(y_test, reg.predict(x_test)))
reg.score(x_test,y_test)

df=pd.get_dummies(data,columns=['waterfront','view','condition','grade','zipcode'], drop_first=True)
y=data['price']
df= df.drop(['date','id','price'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,train_size=0.8,random_state=42)

reg.fit(x_train,y_train)

print('Coefficients: \n', reg.coef_)
print(metrics.mean_squared_error(y_test, reg.predict(x_test)))
print(reg.score(x_test,y_test))