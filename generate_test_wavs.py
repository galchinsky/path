import os
import numpy as np
import soundfile as sf


def frequency_for_index(idx, base_freq=55.0):
    """Returns frequency shifted up by idx semitones from base (A1 = 55 Hz)"""
    return base_freq * (2 ** (idx / 12.0))


def generate_seamless_sine(freq, sample_rate=44100, cycles=256, amplitude=0.1):
    """Generates stereo sine wave completing exact number of cycles for clean looping"""
    period = 1.0 / freq
    duration = cycles * period
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    stereo_wave = np.stack([wave, wave], axis=-1)
    return stereo_wave, sample_rate


def main(output_dir="loops"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[WAV GEN] Saving files to: {output_dir}")

    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            freq = frequency_for_index(idx)
            wave, sr = generate_seamless_sine(freq)
            filename = os.path.join(output_dir, f"{i}_{j}.wav")
            sf.write(filename, wave, samplerate=sr)
            print(f"[WAV GEN] {filename} - {freq:.2f} Hz")

    print("[WAV GEN] Done.")


if __name__ == "__main__":
    main()
