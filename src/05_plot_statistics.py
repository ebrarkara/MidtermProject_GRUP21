import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Verileri Oku
file_path = "outputs/features.xlsx"
if not os.path.exists(file_path):
    print(f"❌ HATA: {file_path} bulunamadı! Önce 02_extract_features.py çalıştır.")
    exit()

df = pd.read_excel(file_path)

# 2. Veri Temizleme (F0 değeri 0 veya NaN olanları analiz dışı bırakıyoruz)
df_clean = df[df["Mean_F0"] > 0].copy()

# Cinsiyet etiketlerini rapor için güzelleştirme (E -> Erkek, K -> Kadın, C -> Çocuk)
label_map = {'E': 'Erkek', 'K': 'Kadın', 'C': 'Çocuk'}
df_clean['Cinsiyet_Etiket'] = df_clean['Cinsiyet'].map(label_map)

# İSTATİSTİKSEL TABLO (Ödev Madde 5)python src/05_plot_statistics.py
# Sınıf bazlı ortalama, standart sapma ve sayıları hesaplıyoruz
stats_table = df_clean.groupby('Cinsiyet_Etiket')['Mean_F0'].agg(['count', 'mean', 'std']).reset_index()
stats_table.columns = ['Sınıf', 'Örnek Sayısı', 'Ortalama F0 (Hz)', 'Standart Sapma']

print("\n📊 --- HOCANIN İSTEDİĞİ İSTATİSTİK TABLOSU ---")
print(stats_table)

# Tabloyu kaydet 
os.makedirs("outputs", exist_ok=True)
stats_table.to_excel("outputs/statistics_summary.xlsx", index=False)

# 4. GÖRSELLEŞTİRME (BOXPLOT) - Ödev Madde 6/4
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Boxplot: Sınıflar arası F0 dağılımını gösterir
sns.boxplot(x='Cinsiyet_Etiket', y='Mean_F0', data=df_clean, palette="Set2", order=['Erkek', 'Kadın', 'Çocuk'])

plt.title("Sınıflara Göre Temel Frekans (F0) Dağılımı", fontsize=14)
plt.xlabel("Sınıf (Cinsiyet)", fontsize=12)
plt.ylabel("Ortalama F0 (Hz)", fontsize=12)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/f0_boxplot_final.png", dpi=300)
print("\n✅ Grafik kaydedildi: plots/f0_boxplot_final.png")
plt.show()