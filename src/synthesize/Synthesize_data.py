import torch
import torch.nn.functional as F
import torchaudio
from scipy import signal
import numpy as np
import os
from pathlib import Path

def BandlimitedNoise_generation(f_start, Bandwidth, fs, N):
    #音声時間 = N / fs
    numtaps = 1024
    f_end = f_start + Bandwidth
    b2 = signal.firwin(numtaps, [f_start, f_end], pass_zero="bandpass", window="hamming", fs=fs)
    xin = np.random.randn(N + numtaps-1)
    Re = signal.lfilter(b2, 1, xin)
    Noise = Re[numtaps-1:]
    Noise = Noise / np.max(np.abs(Noise))
    return Noise

def BandlimitFilter_generation(f_starts, Bandwidthes, fs, type=np.float32):
    #return [batch, numtaps]: torch.tensor
    numtaps = 1024
    f_ends = f_starts + Bandwidthes
    filters = torch.tensor([signal.firwin(numtaps, [f_start, f_end], pass_zero="bandpass", window="hamming", fs=fs).astype(type) for f_start, f_end in zip(f_starts, f_ends)], pin_memory=True)
    filters = torch.flip(filters, [-1])
    return filters

class SoundGenerator():
    
    def __init__(self, min, max, fs, time, folder, type=np.float32):
        self.min = min
        self.max = max
        self.fs = fs
        self.len = fs * time
        self.folder = Path(folder)
        self.Num = 1
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.folder.mkdir(parents=True, exist_ok=True)
        
    def _construct_(self):
        f_star = int(np.random.uniform(self.min, self.max))
        bandWidth = int(np.random.uniform(1,self.max - f_star))
        f_end = f_star + bandWidth 
        filename = f'{self.Num}_from_{f_star}_to_{f_end}_Hz.wav'
        filePath = os.path.join(self.folder, filename)
        noise = BandlimitedNoise_generation(f_star, bandWidth, self.fs, self.len) 
        noise = torch.from_numpy(noise).type(torch.float32).unsqueeze(0)
        torchaudio.save(filePath, noise, self.fs)
        self.Num += 1 
        return f_star, f_end, filename
    
    def _fast_construct_(self, batch):
        f_star = np.random.uniform(self.min, self.max, batch).astype(int)
        bandwidth = np.random.uniform(1, self.max - f_star, batch).astype(int)
        f_end = f_star + bandwidth
        filenames = [f"{self.Num + i}_from_{f_star[i]}_to_{f_end[i]}_Hz.wav" for i in range(batch)]
        filters = BandlimitFilter_generation(f_star, bandwidth, self.fs).to(self.device).unsqueeze(1)
        xins = torch.randn([batch, (self.len + 1023)], dtype=torch.float32, device=self.device).unsqueeze(0)
        Noise = F.conv1d(xins, filters, groups=batch, stride=1, padding=0).squeeze()
        Noise = Noise / torch.max(torch.abs(Noise), dim=1, keepdim=True).values
        Noise = Noise.cpu()
        from concurrent.futures import ThreadPoolExecutor
        def save_worker(args):
            name, noise = args
            filePath = os.path.join(self.folder, name)
            torchaudio.save(filePath, noise.unsqueeze(0), self.fs)
        tasks = list(zip(filenames, Noise))
        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(save_worker, tasks)
        # for name, noise in zip(filenames, Noise):
        #     filePath = os.path.join(self.folder, name)
        #     torchaudio.save(filePath, noise.unsqueeze(0), self.fs)
        self.Num += batch