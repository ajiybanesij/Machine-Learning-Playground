import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Verilerin Yüklenmesi
    Pandas ile csv formatındaki veriyi bir değişkene aktarıyor."""
datas = pd.read_csv(r"veriler.csv")

"""
Veri Ön İşleme
    Değişkene aktarılmış tabloyu sütünlarına ulaşmak."""
height = datas[['boy']]
# Çıktı => index,boy listesi

height_weight = datas[['boy', 'kilo']]
# Çıktı => index,boy,kilo listesi
