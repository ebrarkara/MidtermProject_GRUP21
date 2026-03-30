import numpy as np

def frame_signal(signal, frame_size, hop_size):
    frames = []
    for start in range(0, len(signal) - frame_size, hop_size):
        frame = signal[start:start + frame_size]
        frames.append(frame)
    return np.array(frames)

def compute_energy(frame):
    return np.sum(frame ** 2)

def compute_zcr(frame):
    zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return zero_crossings / len(frame)

def detect_voiced_frames(energies, zcrs, energy_threshold_ratio=0.1, zcr_threshold=0.15):
    energy_threshold = np.max(energies) * energy_threshold_ratio
    voiced_flags = (energies > energy_threshold) & (zcrs < zcr_threshold)
    return voiced_flags

def autocorrelation_pitch(frame, sr, fmin=80, fmax=400):
    frame = frame - np.mean(frame)

    if np.all(frame == 0):
        return 0

    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    if max_lag >= len(autocorr):
        return 0

    search_region = autocorr[min_lag:max_lag]

    if len(search_region) == 0:
        return 0

    peak_index = np.argmax(search_region)
    peak_lag = peak_index + min_lag

    if peak_lag == 0:
        return 0

    f0 = sr / peak_lag
    return f0