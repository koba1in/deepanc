import torch
import os
import numpy as np
from torch import nn
import scipy.signal as signal
import scipy.io as sio

from common.Network import crn_model
from common.utils import stft, spec_to_antinoise

def load_weight_for_model(model, pretrained_path, device):
    model_dict = model.state_dict()
    pretrained_dict = {k.replace("module.", "", 1): v for k, v in torch.load(pretrained_path, map_location=device).items()}
    print(model_dict.keys())
    print(pretrained_dict.keys())
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)
    
def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data)/(max-min)

class Deep_ANC_controller():
    
    def __init__(self, MODEL_PATH, device, fs, delay, eta, secondary_path):
        model = crn_model
        load_weight_for_model(model, MODEL_PATH, device)
        model = model.to(device)
        model.eval()
        
        self.device = device
        self.model = model
        self.fs = fs
        self.delay = delay
        if eta.ndim == 1:
            eta = eta.unsqueeze(0)
        self.eta = eta.to(device)
        for i in range(1, 3):
            if secondary_path.ndim == i:
                secondary_path = secondary_path.unsqueeze(0)
        self.sec_path = secondary_path.to(device)
        
    def noise_cancellation(self, Dis, waveform):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        Dis = Dis.to(self.device)
        complex_spectrogram = stft(waveform, self.delay)
        complex_spectrogram = self.model(complex_spectrogram)
        antinoise = spec_to_antinoise(complex_spectrogram, self.sec_path, self.eta)
        e = Dis - antinoise
        return e.tolist()

