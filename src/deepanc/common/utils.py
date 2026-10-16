import torch
import torch.nn.functional as F
import math

N_FFT = 320
HOP_LENGTH = 160


def stft(wave, delay):
    #[Batch, T] -> [Batch, 2, T, 161]
    window = torch.hamming_window(N_FFT).to(wave.device)
    stft = torch.stft(wave, N_FFT, HOP_LENGTH, window=window, center=False, return_complex=True)
    stft = torch.view_as_real(stft)
    T = stft.size(2)
    stft = F.pad(stft, (0, 0, delay, 0, 0, 0, 0, 0))
    stft = stft[:, :, :T, :]
    stft = torch.permute(stft, (0, 3, 2, 1))
    return stft

def istft(stft):
    #[Batch, 2, T, 161] -> [Batch, time]
    window = torch.hamming_window(N_FFT).to(stft.device)
    stft = torch.permute(stft, (0, 3, 2, 1)).contiguous()
    stft_complex = torch.view_as_complex(stft)
    wave = torch.istft(stft_complex, N_FFT, HOP_LENGTH, window=window, center=False,)
    return wave
    
# def fls(wave, etas):
#     #wave: [Batch, T]
#     #etas: [Batch, 1]
#     return math.sqrt(math.pi / 2) * (etas * torch.erf(wave / (math.sqrt(2) * etas)))
def fls(wave, etas):
    # wave: [Batch, T]
    # etas: [Batch, 1]
    linear_mask = torch.isinf(etas) | (etas >= 1e10)

    safe_etas = torch.where(
        linear_mask,
        torch.ones_like(etas),
        etas.clamp_min(torch.finfo(wave.dtype).eps),
    )

    result = (
        math.sqrt(math.pi / 2)
        * safe_etas
        * torch.erf(wave / (math.sqrt(2) * safe_etas))
    )


    return torch.where(linear_mask, wave, result)

def spec_to_antinoise(spec, Sec_pathes, etas):
    #spec: [batch, 2, T, 161]
    #Sec_pathes: [batch, 1, numtaps]
    #etas: [batch, 1]
    wave = istft(spec)
    wave = fls(wave, etas)
    wave = apply_impulse_response(wave, Sec_pathes)
    return wave

def apply_impulse_response(wave, pathes):
    #wave: [batch, time]
    #pathes: [batch, 1, numtaps]
    batch_size = wave.shape[0]
    wave = wave.unsqueeze(0)
    wave = F.pad(wave, (pathes.shape[2]-1, 0), "constant", 0)
    pathes = pathes.flip(-1)
    wave = F.conv1d(wave, pathes, groups=batch_size)
    wave = wave.squeeze(0)
    return wave

def loss(disturbance, antinoise):
    #[Batch, T]
    return (disturbance-antinoise).square().mean()


def nmse(err, dis):    
    # torch.mean(10 * torch.log10(torch.sum(torch.pow(err, 2))/torch.sum(torch.pow(dis, 2))))
    numerator = err.square().sum()
    denominator = dis.square().sum().clamp_min(1e-12)
    return 10 * torch.log10(numerator / denominator)


import os
import scipy.io as sio
import numpy as np
def loading_paths_from_MAT(folder, subfolder, Pri_Path, Sec_Path, Pri_path_name="Pz1", Sec_path_name="S"):
    Pri_Path_file, Sec_Path_file = os.path.join(folder, subfolder, Pri_Path), os.path.join(folder, subfolder, Sec_Path)
    Pri_dfs, Sec_dfs = sio.loadmat(Pri_Path_file), sio.loadmat(Sec_Path_file)
    Pri_path, Sec_path = Pri_dfs[Pri_path_name].squeeze(), Sec_dfs[Sec_path_name].squeeze()
    Pri_path, Sec_path = Pri_path.astype(np.float32), Sec_path.astype(np.float32)
    return [Pri_path, Sec_path]


def check_finite(name, x):
    print(
        name,
        "finite =", torch.isfinite(x).all().item(),
        "min =", x.nan_to_num().min().item(),
        "max =", x.nan_to_num().max().item(),
        "mean =", x.nan_to_num().mean().item(),
    )