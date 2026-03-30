import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.fft import rfft, rfftfreq

# 1. VERİ SETİNDEKİ İLK SES DOSYASINI OTOMATİK BUL 
search_path = "Dataset/**/*.wav"
wav_files = glob.glob(search_path, recursive=True)

if not wav_files:
    print("❌ HATA: Dataset içinde .wav dosyası bulunamadı!")
    exit()

# Analiz için ilk dosyayı seçiyoruz
audio_path = wav_files[0]
print(f"✅ Analiz Edilen Denek Ses: {audio_path}")

# 2. SESİ YÜKLE VE PENCERELE
signal, sr = librosa.load(audio_path, sr=None)

# 30ms'lik bir parça alıyoruz (Durağan bölge tespiti için orta kısımdan)
frame_duration = 0.030 
frame_size = int(frame_duration * sr)
start_idx = len(signal) // 3
frame = signal[start_idx : start_idx + frame_size]
frame = frame - np.mean(frame) # DC Ofset temizleme

# --- OTOKORELASYON (Zaman Düzlemi) ---
autocorr = np.correlate(frame, frame, mode='full')
autocorr = autocorr[len(autocorr)//2:]
lags = np.arange(len(autocorr)) 

# Otokorelasyon üzerinden F0 bulma
min_lag = int(sr / 450) # 450 Hz
max_lag = int(sr / 80)  # 80 Hz
search_region = autocorr[min_lag:max_lag]
peak_lag = np.argmax(search_region) + min_lag
f0_autocorr = sr / peak_lag

# --- FFT (Frekans Düzlemi) ---
N = len(frame)
fft_magnitude = np.abs(rfft(frame))
freqs = rfftfreq(N, d=1/sr)

# FFT üzerinden en yüksek tepenin frekansını bulma
peak_freq_idx = np.argmax(fft_magnitude)
f0_fft = freqs[peak_freq_idx]

# --- GÖRSEL KIYASLAMA  ---
plt.figure(figsize=(14, 6))

# Sol Grafik: Otokorelasyon
plt.subplot(1, 2, 1)
plt.plot(lags, autocorr)
plt.axvline(x=peak_lag, color='r', linestyle='--', label=f'Tepe (Lag): {peak_lag}')
plt.title(f"Otokorelasyon (F0 ≈ {f0_autocorr:.2f} Hz)")
plt.xlabel("Gecikme (Lag)")
plt.ylabel("R(tau)")
plt.legend()
plt.grid(True)

# Sağ Grafik: FFT Spektrumu
plt.subplot(1, 2, 2)
plt.plot(freqs, fft_magnitude)
plt.axvline(x=f0_fft, color='r', linestyle='--', label=f'F0 Tepesi: {f0_fft:.2f} Hz')
plt.title(f"FFT Spektrumu (Tepe ≈ {f0_fft:.2f} Hz)")
plt.xlabel("Frekans (Hz)")
plt.ylabel("Genlik")
plt.xlim(0, 1000) # İnsan sesi için 1000Hz yeterli
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/autocorr_vs_fft.png", dpi=300)
print(f"🎯 Otokorelasyon F0: {f0_autocorr:.2f} Hz")
print(f"🎯 FFT Peak: {f0_fft:.2f} Hz")
print(f"🚀 Grafik kaydedildi: plots/autocorr_vs_fft.png")
plt.show()