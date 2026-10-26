from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import csv

class MyDataset(Dataset):

    def __init__(self, audio_folder, csv_path=None):
        if csv_path is None:
            self.audio_files = [ i for i in Path(audio_folder).iterdir() if i.is_file()]
        else:
            with open(csv_path, mode="r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                self.audio_files = [ audio_folder / file[0] for file in reader if file[0].endswith((".wav", ".flac"))]
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        signal,_ = torchaudio.load(self.audio_files[index])
        return signal
    
