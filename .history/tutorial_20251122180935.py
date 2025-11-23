import soundfile as sf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_spectrogram_and_save(signal, sample_rate, output_path: Path):
    if signal.ndim > 1:
        # signal is (num_samples, num_channels); transpose to (channels, samples)
        signal = librosa.to_mono(signal.T)  # now 1D
        
    # "naive" downsample to 11025 Hz for faster processing and noise reduction
    # equivalent to a f_max of ~5kHz
    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=11025)
    sample_rate = 11025

    stft = librosa.stft(signal, n_fft = 2048)
    print(f"STFT shape: {stft.shape}")  # (n_fft/2 + 1, num_frames)
    spectorgram = np.abs(stft) # since stft is a complex number, we take the magnitude (real part))
    log_spectorgram = librosa.amplitude_to_db(spectorgram)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectorgram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.savefig(output_path)
    plt.close()

def main():
    signal, sample_rate = sf.read(Path('data') / 'osakablues.flac')
    print(f"Sample Rate: {sample_rate}")
    plot_spectrogram_and_save(signal, sample_rate, Path('imgs') / 'spectrogram.png')

if __name__ == '__main__':
    main()