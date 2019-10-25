import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

"""Verilerin Yüklenmesi"""
datas = pd.read_csv(r"eksikveriler.csv")

"""Eksik Verilerin Düzeltilmesi
    Imputer ile eksik verinin nasıl doldurulacağının stratejisi belirleniyor.
        Parametreler
        missing_values='NaN'  => eksik verinin NaN şeklinde bulunduğunu belirtiyor.
        strategy='mean'       => eksik verinin Ortalama(Mean) ile bulunması gerektiğini belirtiyor.
        axis=0                => eksik verinin satırda veya sütünda olduğunu belirtiyor.0 => sütün, 1 => satır."""
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

""" illoc ile hangi kolonların seçildiği belirtiliyor.
    [:,1:4].values            => 1. ile 4. kolonlar arasındaki kolonların değerlerini ageData değişkenine ata."""
ageData = datas.iloc[:, 1:4].values

print("Eksik Verili Data")
print(ageData)
print("# # # # # # # # # # # # # # # # # # # # ")


"""fit() ile yukarıda belirlenen stratejisi veri üzerinde uygulanır"""
imputer = imputer.fit(ageData[:, 1:4])

"""transform() ile strateji uygulanmış veriler yeniden ageData değişkenine atılır ve eksik veriler doldurulmuş olur."""
ageData[:, 1:4] = imputer.transform(ageData[:, 1:4])

print("Eksik Veri Doldurulmuş Data")
print(ageData)
print("# # # # # # # # # # # # # # # # # # # # ")