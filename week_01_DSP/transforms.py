from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        # Your code here
        # wav_len = waveform.shape()

        pad = np.zeros(self.window_size // 2)
        wav_padded = np.concatenate((pad, waveform, pad))
        
        windows=[]

        for i in range((len(waveform) - self.window_size % 2) // self.hop_length + 1):
            lef_index = i * self.hop_length
            right_index = lef_index + self.window_size
                         
            windows.append(wav_padded[lef_index:right_index])
    
        windows = np.stack(windows)
        # ^^^^^^^^^^^^^^

        return windows
    

class Hann:
    def __init__(self, window_size=1024):
        # Your code here
        self.weights = scipy.signal.windows.hann(window_size, sym=False)
        # ^^^^^^^^^^^^^^

    
    def __call__(self, windows):
        # Your code here
        return self.weights * windows
        # ^^^^^^^^^^^^^^



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        # Your code here
        freqs = np.fft.rfft(windows)
        freqs = np.abs(freqs)
        if self.n_freqs is not None:
            freqs = freqs[:, :self.n_freqs]

        spec = freqs
        # ^^^^^^^^^^^^^^

        return spec


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        # Your code here
        self.mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=1, fmax=8192)
        # print(f"{self.mel.shape=}")
        self.inverse_mel = np.linalg.pinv(self.mel)
        # print(f"{self.inverse_mel.shape=}")
        # ^^^^^^^^^^^^^^


    def __call__(self, spec):
        # Your code here
        mel = spec @ self.mel.T
        # ^^^^^^^^^^^^^^

        return mel

    def restore(self, mel):
        # Your code here
        spec = mel @ self.inverse_mel.T
        # ^^^^^^^^^^^^^^

        return np.maximum(0, spec)


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        # Your code here
        return mel[::-1, ]
        # ^^^^^^^^^^^^^^



class Loudness:
    def __init__(self, loudness_factor):
        # Your code here
        self.loudness_factor = loudness_factor
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        return mel * self.loudness_factor
        # ^^^^^^^^^^^^^^




class PitchUp:
    def __init__(self, num_mels_up):
        # Your code here
        self.num_mels_up = num_mels_up
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        pitched_mel = np.zeros_like(mel)
        pitched_mel[:, self.num_mels_up:] = mel[:, :-self.num_mels_up]

        return pitched_mel
        # ^^^^^^^^^^^^^^



class PitchDown:
    def __init__(self, num_mels_down):
        # Your code here
        self.num_mels_down = num_mels_down
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        pitched_mel = np.zeros_like(mel)
        pitched_mel[:, :-self.num_mels_down] = mel[:, self.num_mels_down:]

        return pitched_mel
        # ^^^^^^^^^^^^^^



class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        # Your code here
        self.speed_up_factor = speed_up_factor
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        source_indices = np.linspace(0, mel.shape[0] - 1, int(mel.shape[0] * self.speed_up_factor))
        source_indices = np.round(source_indices).astype(int)

        mel = mel[source_indices]
        
        return mel
        # ^^^^^^^^^^^^^^



class FrequenciesSwap:
    def __call__(self, mel):
        # Your code here
        return mel[:, ::-1]
        # ^^^^^^^^^^^^^^



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        # Your code here
        self.quantile = quantile
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        threshold = np.quantile(mel, self.quantile)
        new_mel = mel.copy()
        new_mel[new_mel < threshold] = 0
        
        return new_mel
        # ^^^^^^^^^^^^^^


class Cringe1:
    def __init__(self, rate: float = 5.0, depth: float = 0.8):
        # Your code here
        self.rate = rate
        self.depth = np.clip(depth, 0, 1)
        # ^^^^^^^^^^^^^^

    def __call__(self, mel):
        # Your code here
        n_frames = mel.shape[0]
        
        time_axis = np.linspace(0, self.rate * 2 * np.pi, n_frames)
        oscillator = np.sin(time_axis)
        modulation = (oscillator + 1) / 2
        
        modulation = 1 - self.depth * modulation

        return mel * modulation[:, np.newaxis]
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        self.time_reverser = TimeReverse()
        self.freq_swapper = FrequenciesSwap()
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        reversed_mel = self.time_reverser(mel)
        swapped_mel = self.freq_swapper(reversed_mel)
        
        return swapped_mel
        # ^^^^^^^^^^^^^^

