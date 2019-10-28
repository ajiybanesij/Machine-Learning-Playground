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

"""Ülke verilerinin Dataframe haline getirilmesi"""
sonuc=pd.DataFrame(data=countryData,index=range(22),columns =['fr','tr','us'])

"""ageData verilerinin Dataframe haline getirilmesi"""
sonuc2=pd.DataFrame(data=ageData,index=range(22),columns=['boy','kilo','ageData'])

"""Veriden cinsiyet verilerinin alınması"""
cinsiyet=datas.iloc[:,-1].values

"""cinsiyet verilerini Dataframe haline getirilmesi"""
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

"""Dataframelerinin birleştirilmesi"""
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)


""""Test ve Train Verisi Ayırma"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)