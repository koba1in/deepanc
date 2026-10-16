from torch.utils.data import Dataset
import torchaudio
from pathlib import Path

def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data)/(max-min)

class MyDataset(Dataset):

    def __init__(self, sig_folder,):
        self.sig_files = [ i for i in Path(sig_folder).iterdir() if i.is_file()]
        
    
    def __len__(self):
        return len(self.sig_files)

    def __getitem__(self, index):
        signal,_ = torchaudio.load(self.sig_files[index])
        return signal
    
