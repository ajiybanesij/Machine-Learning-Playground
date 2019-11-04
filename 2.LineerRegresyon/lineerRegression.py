from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler
from sklearn.model_selection import train_test_split

"""Verilerin Yüklenmesi"""
datas = pd.read_csv(r"2.LineerRegresyon/satislar.csv")

"""Veri Ön İşleme"""
aylar = datas[['Aylar']]
satislar = datas[['Satislar']]

"""Verilerin Test ve Train Olarak Bölünmesi"""
x_train, x_test, y_train, y_test = train_test_split(
    aylar, satislar, test_size=0.33, random_state=0)

"""Verilerin Standartlaştırılması
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
"""LineerRegresyon"""
lr = LinearRegression()

"""Test Verilerini Kullanarak Modeli İnşa Etmeye Çalışıyor"""
lr.fit(x_train,y_train)

"""Tahmin"""
tahmin = lr.predict(x_test)


"""Verileri Index'e Göre Sıralama"""
x_train=x_train.sort_index()
y_train=y_train.sort_index()

"""Verileri Görselleştirme"""
plt.plot(x_train,y_train)   #Gerçek verilerin görselleştirilmesi
plt.plot(x_test,tahmin)     #Tahmin verilerinin gösterilmesi

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Saışlar")
plt.show()