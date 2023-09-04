import math, random
import torch
import torchaudio
from torchaudio import transforms

class AudioUtil():
    def loadWav(audio_file):
        sig, sr = torchaudio.load(audio_file)
        print("loadWav")
        return (sig, sr)

    def rechannel(aud, new_channel):
        sig, sr = aud

        if sr.shape[0] == new_channel:
            return aud
        
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    def resample(aud, newsr):
        sig, sr = aud

        if sr == newsr:
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//600 * max_ms

        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        spec = transforms.AmplitudeToDB(top_db=top_db)(sig)

        return (spec)

if __name__ == "__main__":
    print("test")