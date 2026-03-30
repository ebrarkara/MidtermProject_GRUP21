import os
import re
import unicodedata
import numpy as np
import pandas as pd
import librosa

# =========================
# AYARLAR
# =========================
# Master metadata dosyası (tüm excel’lerin birleşmiş hali)
MASTER_EXCEL = "outputs/master_metadata.xlsx"

# Çıkış dosyaları (özelliklerin kaydedileceği yerler)
OUTPUT_EXCEL = "outputs/features.xlsx"
OUTPUT_CSV = "outputs/features.csv"

# Dataset klasörünün kök yolu
DATASET_ROOT = "Dataset"

# Frame ve hop boyutları (ms cinsinden)
FRAME_MS = 25   # pencere boyutu (25 ms)
HOP_MS = 10     # kayma miktarı (10 ms)


# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def normalize_text(text):
    """Türkçe karakterleri, boşlukları ve uzantı sorunlarını normalize eder."""
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Unicode normalize (karakterleri sadeleştirir)
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])

    # Türkçe karakter dönüşümleri
    replacements = {
        "ı": "i", "İ": "I",
        "ş": "s", "Ş": "S",
        "ğ": "g", "Ğ": "G",
        "ü": "u", "Ü": "U",
        "ö": "o", "Ö": "O",
        "ç": "c", "Ç": "C",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # boşlukları alt çizgi yap
    text = text.replace(" ", "_")
    return text


def frame_signal(signal, frame_size, hop_size):
    # Sinyali küçük parçalara (frame) böler
    frames = []
    for start in range(0, len(signal) - frame_size + 1, hop_size):
        frame = signal[start:start + frame_size]
        frames.append(frame)
    return np.array(frames)


def compute_energy(frame):
    # Kısa süreli enerji hesaplama
    return np.sum(frame ** 2)


def compute_zcr(frame):
    # Sıfır geçiş oranı hesaplama
    zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return zero_crossings / len(frame)


def detect_voiced_frames(energies, zcrs, energy_threshold_ratio=0.1, zcr_threshold=0.15):
    # Sesli (voiced) frame’leri tespit eder
    if len(energies) == 0:
        return np.array([])

    # enerji eşik değeri (max enerjiye göre)
    energy_threshold = np.max(energies) * energy_threshold_ratio

    # voiced = yüksek enerji + düşük zcr
    voiced_flags = (energies > energy_threshold) & (zcrs < zcr_threshold)
    return voiced_flags


def autocorrelation_pitch(frame, sr, fmin=80, fmax=450):
    # Otokorelasyon ile F0 hesaplama

    # ortalamayı çıkar (DC offset kaldırma)
    frame = frame - np.mean(frame)

    if np.all(frame == 0):
        return 0

    # otokorelasyon hesapla
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # gecikme sınırları (pitch aralığı)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    if max_lag >= len(autocorr) or min_lag >= max_lag:
        return 0

    # uygun aralıkta en yüksek tepeyi bul
    search_region = autocorr[min_lag:max_lag]

    if len(search_region) == 0:
        return 0

    peak_index = np.argmax(search_region)
    peak_lag = peak_index + min_lag

    if peak_lag == 0:
        return 0

    # frekans hesapla
    f0 = sr / peak_lag
    return f0


def find_audio_file(filename, dataset_root=DATASET_ROOT):
    """
    Dataset içinde dosyayı arar.
    Türkçe karakter ve uzantı problemlerini tolere eder.
    """
    if pd.isna(filename):
        return None

    raw_name = str(filename).strip()

    # uzantı yoksa ekle
    if not raw_name.lower().endswith(".wav"):
        raw_name += ".wav"

    target_norm = normalize_text(raw_name).lower()

    # tüm klasörlerde ara
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            f_norm = normalize_text(f).lower()
            if f_norm == target_norm:
                return os.path.join(root, f)

    return None


def extract_features(audio_path):
    # ses dosyasını yükle
    signal, sr = librosa.load(audio_path, sr=None)

    if len(signal) == 0:
        return None

    # frame boyutlarını hesapla
    frame_size = int(sr * FRAME_MS / 1000)
    hop_size = int(sr * HOP_MS / 1000)

    if len(signal) < frame_size:
        return None

    # sinyali framelere böl
    frames = frame_signal(signal, frame_size, hop_size)

    if len(frames) == 0:
        return None

    # her frame için enerji ve zcr hesapla
    energies = np.array([compute_energy(frame) for frame in frames])
    zcrs = np.array([compute_zcr(frame) for frame in frames])

    # voiced frame’leri bul
    voiced_flags = detect_voiced_frames(energies, zcrs)

    voiced_f0s = []
    voiced_energies = []
    voiced_zcrs = []

    # sadece voiced frame’lerde işlem yap
    for i, frame in enumerate(frames):
        if i < len(voiced_flags) and voiced_flags[i]:
            f0 = autocorrelation_pitch(frame, sr)

            # geçerli F0 aralığı
            if 80 <= f0 <= 450:
                voiced_f0s.append(f0)
                voiced_energies.append(energies[i])
                voiced_zcrs.append(zcrs[i])

    if len(voiced_f0s) == 0:
        return None

    # ortalama değerleri döndür
    return {
        "Mean_F0": round(float(np.mean(voiced_f0s)), 2),
        "Std_F0": round(float(np.std(voiced_f0s)), 2),
        "Mean_Energy": round(float(np.mean(voiced_energies)), 6),
        "Mean_ZCR": round(float(np.mean(voiced_zcrs)), 6),
        "Voiced_Frame_Count": int(len(voiced_f0s))
    }


# =========================
# ANA İŞLEM
# =========================
def main():
    # master excel var mı kontrol et
    if not os.path.exists(MASTER_EXCEL):
        print(f"❌ Master excel bulunamadı: {MASTER_EXCEL}")
        return

    # excel’i oku
    df = pd.read_excel(MASTER_EXCEL)

    # dosya adı kolonunu bul
    possible_file_cols = ["Dosya_Adi", "Dosya Adi", "DosyaAdı", "Dosya", "Filename", "FileName"]
    file_col = None

    for col in possible_file_cols:
        if col in df.columns:
            file_col = col
            break

    if file_col is None:
        print("❌ Dosya adı kolonu bulunamadı!")
        print("Bulunan kolonlar:", list(df.columns))
        return

    print(f"🔍 Toplam {len(df)} kayıt analiz edilecek...\n")

    results = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # her satır için işlem yap
    for idx, row in df.iterrows():
        filename = row[file_col]

        # dosya yolunu bul
        audio_path = find_audio_file(filename)

        if audio_path is None:
            print(f"⚠️ Dosya bulunamadı, atlanıyor: {filename}")
            skipped_count += 1
            continue

        try:
            # özellik çıkar
            feats = extract_features(audio_path)

            if feats is None:
                print(f"⚠️ Özellik çıkarılamadı, atlanıyor: {os.path.basename(audio_path)}")
                skipped_count += 1
                continue

            # yeni satır oluştur
            new_row = row.to_dict()
            new_row["Dosya_Path"] = audio_path
            new_row.update(feats)

            results.append(new_row)
            processed_count += 1

            print(f"✅ [{processed_count}] işlendi -> {os.path.basename(audio_path)}")

        except Exception as e:
            print(f"❌ Hata oluştu: {audio_path} -> {e}")
            error_count += 1
            continue

    # sonuçları dataframe yap
    features_df = pd.DataFrame(results)

    # klasör yoksa oluştur
    os.makedirs("outputs", exist_ok=True)

    # kaydet
    features_df.to_excel(OUTPUT_EXCEL, index=False)
    features_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # sonuçları yazdır
    print("\n" + "="*50)
    print(" FEATURE EXTRACTION TAMAMLANDI!")
    print("="*50)
    print(f"📁 Kaydedildi: {OUTPUT_EXCEL}")
    print(f"📁 Kaydedildi: {OUTPUT_CSV}")
    print(f"📊 Toplam işlenen dosya sayısı: {processed_count}")
    print(f"⚠️ Atlanan dosya sayısı: {skipped_count}")
    print(f"❌ Hatalı dosya sayısı: {error_count}")
    print(f"📋 Final tablo satır sayısı: {len(features_df)}")
    print("="*50)


if __name__ == "__main__":
    main()