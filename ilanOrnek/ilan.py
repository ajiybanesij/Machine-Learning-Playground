import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Verilerin Yüklenmesi"""
ilanRawData = pd.read_csv(r"ilanOrnek\ilanVerileri.csv")

fiyatRaw=ilanRawData.iloc[:,0].values
metrekareRaw=ilanRawData.iloc[:,2].values
mahalleRaw = ilanRawData.iloc[:,1].values

"""OneHotEncoder ile kategorize etme"""
ohe=OneHotEncoder(categorical_features="all")
mahalleRaw=mahalleRaw.reshape(-1,1)
mahalleRaw=ohe.fit_transform(mahalleRaw).toarray()

"""Dataframe Haline Getirme"""
mahalleDF=pd.DataFrame(data=mahalleRaw,index=range(950),columns=['Altındağ','Etimesgut','Keçiöğren','Mamak','Sincan','Yenimahalle','Çankaya'])
metrekareDF=pd.DataFrame(data=metrekareRaw,index=range(950),columns=['metrekare'])
fiyatDF=pd.DataFrame(data=fiyatRaw,index=range(950),columns=['fiyat'])
butunVeriler=pd.concat([mahalleDF,metrekareDF,fiyatDF],axis=1)

""""Test ve Train Verisi Ayırma"""
test_data,train_data=train_test_split(butunVeriler,test_size=0.33,random_state=0)

"""Veri Ölçekleme"""
sc= StandardScaler()
#test_data verisini standartlaştırma uygulama.
standart_test_data=sc.fit_transform(test_data)
#train_data verisini standartlaştırma uygulama
standart_train_data=sc.fit_transform(train_data)

