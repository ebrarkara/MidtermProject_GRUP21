import pandas as pd
import os

# 1. Veriyi Oku
df = pd.read_excel("outputs/features.xlsx")

label_column = "Cinsiyet"

# 2. VERİ TEMİZLEME (Ana sınıfları korur)
# Başındaki sonundaki boşlukları temizle ve metne çevir
df[label_column] = df[label_column].astype(str).str.strip()

# Sadece Erkek, Kadın ve Çocuk sınıflarını filtrele (Çöp verileri at)
df = df[df[label_column].isin(["E", "K", "C"])]

# 3. İSTATİSTİKSEL HESAPLAMA (Madde 5 İçin)
stats = df.groupby(label_column).agg(
    Ornek_Sayisi=("Mean_F0", "count"),
    Ortalama_F0_Hz=("Mean_F0", "mean"),
    Std_F0_Hz=("Mean_F0", "std"),
    Ortalama_ZCR=("Mean_ZCR", "mean"),
    Ortalama_Energy=("Mean_Energy", "mean")
).reset_index()

# 4. KAYDET
os.makedirs("outputs", exist_ok=True)
stats.to_excel("outputs/statistics_table.xlsx", index=False)
stats.to_csv("outputs/statistics_table.csv", index=False, encoding="utf-8-sig")

print("İstatistik tablosu başarıyla temizlendi ve oluşturuldu!\n")
print(stats)