import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,Imputer
from sklearn.model_selection import train_test_split

"""Verilerin Yüklenmesi
    Pandas ile csv formatındaki veriyi bir değişkene aktarıyor."""
datas = pd.read_csv(r"veriler.csv")

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
ageData=datas.iloc[:,1:4].values
imputer=imputer.fit(ageData[:,1:4]) 
ageData[:,1:4]=imputer.transform(ageData[:,1:4])

countryData=datas.iloc[:,0:1].values
le=LabelEncoder()
countryData[:,0]=le.fit_transform(countryData[:,0])
ohe=OneHotEncoder(categorical_features='all')
countryData=ohe.fit_transform(countryData).toarray()
sonuc=pd.DataFrame(data=countryData,index=range(22),columns =['fr','tr','us'])
sonuc2=pd.DataFrame(data=ageData,index=range(22),columns=['boy','kilo','ageData'])
cinsiyet=datas.iloc[:,-1].values
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
s=pd.concat([sonuc,sonuc2],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

"""Standart Ölçekleme Yapma"""
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
#x_test verisini standartlaştırma uygulama.
X_test=sc.fit_transform(x_test)
print(X_test)
print("#####################")
#x_train verisini standartlaştırma uygulama
X_train=sc.fit_transform(x_train)
print(X_train)