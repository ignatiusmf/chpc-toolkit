import torch
import numpy as np
from scipy.stats import kurtosis
import pywt
from scipy.stats import skew
from torchvision.transforms import GaussianBlur
import torch.fft as fft

device = 'cuda'

def compute_wavelet_energy(feature_map, wavelet='db1', level=2):
    energies = []
    for c in range(feature_map.shape[1]):  # Iterate over channels
        fm_np = feature_map[0, c].detach().cpu().numpy()
        coeffs = pywt.wavedec2(fm_np, wavelet, level=level)
        high_freq_energy = 0
        for detail in coeffs[1:]:  # Each detail is a tuple of arrays
            for subband in detail:
                high_freq_energy += np.sum(subband**2)
        energies.append(high_freq_energy)
    
    energies = torch.tensor(energies)
    avg = energies.mean().item()
    return avg

def compute_skewness(feature_map):
    flat_fm = feature_map.flatten().detach().cpu().numpy()
    skewness = skew(flat_fm)
    return skewness

def compute_lowpass_residuals(feature_map, kernel_size=3, sigma=1.0):
    blur = GaussianBlur(kernel_size, sigma)
    denoised_fm = blur(feature_map)
    residuals = torch.norm(feature_map - denoised_fm, p=2)
    return residuals.item()

def compute_frequency_entropy(feature_map):
    channels = feature_map.shape[1]
    entropies = []
    for c in range(channels):
        fm = feature_map[0, c]  
        fm_fft = fft.fft2(fm)
        power_spectrum = torch.abs(fm_fft)**2
        probs = power_spectrum / (power_spectrum.sum() + 1e-10)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        entropies.append(entropy)
    
    entropies = torch.tensor(entropies)
    avg = entropies.mean()
    return avg.item()

def compute_iqr(feature_map):
    flat_fm = feature_map.flatten()
    q75, q25 = torch.quantile(flat_fm, torch.tensor([0.75, 0.25],device=device))
    iqr = q75 - q25
    return iqr.item()

def compute_high_freq_power(feature_map):
    channels = feature_map.shape[1]
    high_freq_powers = []
    for c in range(channels):
        fm = feature_map[0, c] 
        fm_fft = fft.fft2(fm)
        power_spectrum = torch.abs(fm_fft)**2
        high_freq_power = power_spectrum[int(fm.shape[0]/2):, int(fm.shape[1]/2):].sum()
        high_freq_powers.append(high_freq_power)
    
    high_freq_powers = torch.tensor(high_freq_powers)
    avg = high_freq_powers.mean().item()
    return avg

def compute_cv(feature_map):
    mean = torch.mean(feature_map)
    std = torch.std(feature_map)
    cv = mean / (std.abs() + 1e-10)
    return cv.item()

def compute_kurtosis(feature_map):
    flat_fm = feature_map.flatten().detach().cpu().numpy()
    return kurtosis(flat_fm)

def compute_entropy(feature_map): 
    feature_map = feature_map - feature_map.min()  
    sum_fm = feature_map.sum()
    if sum_fm == 0:  # Avoid division by zero
        return torch.tensor(0.0, device=feature_map.device)
    probs = feature_map / sum_fm
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    return entropy.item()

def compute_average(feature_map):
    return feature_map.mean().item()

def compute_std(feature_map):
    return feature_map.std().item()

metrics = { 
    "Average": compute_average,
    "STD": compute_std,
    "Kurtosis": compute_kurtosis,
    "Wavelet High-Frequency Energy": compute_wavelet_energy,
    "Low-Pass Filter Residuals": compute_lowpass_residuals,
    "Frequency Entropy": compute_frequency_entropy,
    "IQR": compute_iqr,
    "High-Frequency Power": compute_high_freq_power,
    "Coefficient of Variation": compute_cv,
    "Skewness": compute_skewness,
    "Total Entropy": compute_entropy,
}
    