from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

from AudioUtil import AudioUtil

class SoundDs(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.iloc[0 + idx - 1, 0]

        map_info = self.df.iloc[0 + idx - 1, 1:]

        aud = AudioUtil.loadWav(audio_file)

        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

        return sgram, map_info