import soundfile as sf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_spectrogram_and_save(signal, sample_rate, output_path: Path):
    stft = librosa.stft(signal, n_fft = 1024)
    spectorgram = np.abs(stft) # since stft is a complex number, we take the magnitude (real part))
    log_spectorgram = librosa.amplitude_to_db(spectorgram)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectorgram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.savefig(output_path)
    plt.close()

def main():
    signal, sample_rate = sf.read(Path('data') / 'foremma.flac')
    print(f"Sample Rate: {sample_rate}")
    plot_spectrogram_and_save(signal, sample_rate, Path('imgs') / 'spectrogram.png')

if __name__ == '__main__':
    main()