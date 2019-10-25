import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


"""Verilerin Yüklenmesi"""
datas=pd.read_csv(r"veriler.csv")

"""Kategorize edilecek verinin alınması"""
country = datas.iloc[:,0:1].values

"""LabelEncode ile kategorize etme"""
le=LabelEncoder()
country[:,0]=le.fit_transform(country[:,0])
#print(country)

"""OneHotEncoder ile kategorize etme
        categorical_features='all' ile bütün hepsini kategorize et"""
ohe=OneHotEncoder(categorical_features="all")
country=ohe.fit_transform(country).toarray()
print(country)