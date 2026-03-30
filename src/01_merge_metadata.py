import pandas as pd
import glob
import os

# Dataset içindeki tüm Excel dosyalarını bul
excel_files = glob.glob("Dataset/**/Grup_*_MetaVeri.xlsx", recursive=True)

if len(excel_files) == 0:
    print("❌ Hiç Excel dosyası bulunamadı!")
    exit()

all_data = []

print("📂 Bulunan Excel dosyaları:")
for file in excel_files:
    print(file)

print("\n📖 Excel dosyaları okunuyor...\n")

for file in excel_files:
    try:
        # Excel oku
        df = pd.read_excel(file)

        # Boş satırları sil
        df = df.dropna(how="all")

        # Sütun isimlerini temizle
        df.columns = [str(col).strip() for col in df.columns]

        # Eğer bazı dosyalarda ilk satır yanlışlıkla başlık değilse düzeltmeye çalış
        if "Dosya_Adi" not in df.columns:
            # İlk satır başlık olabilir diye kontrol et
            first_row = df.iloc[0].astype(str).str.strip().tolist()
            if "Dosya_Adi" in first_row:
                df.columns = first_row
                df = df[1:].reset_index(drop=True)
                df.columns = [str(col).strip() for col in df.columns]

        # Türkçe karakter ve isim karmaşası düzeltmeleri
        rename_map = {
            "Dosya Adi": "Dosya_Adi",
            "Dosya Adı": "Dosya_Adi",
            "Dosya_Adi ": "Dosya_Adi",
            "Dosya_Adi\t": "Dosya_Adi",
            "Yaş": "Yas",
            "Yas ": "Yas",
            "Cinsiyet ": "Cinsiyet",
            "Duygu ": "Duygu",
            "Kayıt_Cihazı": "Kayit_Cihazi",
            "Kayıt Cihazı": "Kayit_Cihazi",
            "Kayit Cihazi": "Kayit_Cihazi",
            "gürültü seviyesi": "gurultu_seviyesi",
            "Gürültü Seviyesi": "gurultu_seviyesi",
            "gürültü_seviyesi": "gurultu_seviyesi",
            "Gurultu Seviyesi": "gurultu_seviyesi"
        }

        df = df.rename(columns=rename_map)

        # Gerekli kolonları garanti altına al
        needed_cols = [
            "Dosya_Adi",
            "Denek_ID",
            "Cinsiyet",
            "Yas",
            "Duygu",
            "Cumle_No",
            "Kayit_Cihazi",
            "ORTAM",
            "gurultu_seviyesi"
        ]

        for col in needed_cols:
            if col not in df.columns:
                df[col] = None

        # Sadece gerekli kolonları al
        df = df[needed_cols]

        # Boşluk temizliği
        for col in ["Dosya_Adi", "Denek_ID", "Cinsiyet", "Duygu", "Kayit_Cihazi", "ORTAM", "gurultu_seviyesi"]:
            df[col] = df[col].astype(str).str.strip()

        # Grup klasörünü al
        group_folder = os.path.dirname(file)

        # Tam ses dosyası yolu oluştur
        df["Dosya_Path"] = df["Dosya_Adi"].apply(lambda x: os.path.join(group_folder, str(x)))

        # Ek bilgi kolonları
        df["Kaynak_Excel"] = file
        df["Grup_Klasoru"] = os.path.basename(group_folder)

        # Tamamen boş / saçma satırları temizle
        df = df[df["Dosya_Adi"].notna()]
        df = df[df["Dosya_Adi"] != ""]
        df = df[df["Dosya_Adi"] != "nan"]

        all_data.append(df)

        print(f"✅ OKUNDU: {file} -> {len(df)} satır")

    except Exception as e:
        print(f"❌ HATA: {file} okunamadı -> {e}")

# Hiç veri okunmadıysa
if len(all_data) == 0:
    print("❌ Hiçbir Excel başarıyla okunamadı!")
    exit()

# Hepsini birleştir
master_df = pd.concat(all_data, ignore_index=True)

# Cinsiyet temizliği
master_df["Cinsiyet"] = master_df["Cinsiyet"].astype(str).str.strip()

# Çıktı klasörü
os.makedirs("outputs", exist_ok=True)

# Kaydet
master_df.to_excel("outputs/master_metadata.xlsx", index=False)
master_df.to_csv("outputs/master_metadata.csv", index=False, encoding="utf-8-sig")

print("\n==============================")
print("MASTER METADATA OLUŞTURULDU!")
print("==============================")
print(f"📁 Toplam Excel sayısı: {len(excel_files)}")
print(f"📄 Toplam kayıt (satır) sayısı: {len(master_df)}")
print(f"🧾 Toplam kolon sayısı: {len(master_df.columns)}")

print("\n📌 Kolonlar:")
print(master_df.columns.tolist())

print("\n📌 İlk 5 satır:")
print(master_df.head())