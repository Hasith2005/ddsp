import torch
import torch.nn.functional as F

def filtered_noise_synth(
    noise_magnitudes: torch.Tensor,  # Shape: (batch, n_frames, 65)
    n_samples: int = 64000,
    hop_length: int = 256,
    n_fft: int = 1024
):
    batch_size, n_frames, _ = noise_magnitudes.shape
    device = noise_magnitudes.device
    ir = torch.fft.irfft(noise_magnitudes, dim=-1)  # (batch, 250, 128)
    window = torch.hann_window(128, device=device)
    ir = torch.roll(ir, shifts=64, dims=-1)
    ir = ir * window
    ir = torch.roll(ir, shifts=-64, dims=-1)

    # Pad from 128 to 1024 to match the STFT size of noise
    ir_padded = F.pad(ir, (0, n_fft - 128))
    
    # RFFT returns n_fft // 2 + 1 bins (513 bins)
    H = torch.fft.rfft(ir_padded, dim=-1)  # (batch, 250, 513)

    # Transpose to match STFT shape: (batch, freq_bins, time_frames)
    H = H.transpose(1, 2)  # (batch, 513, 250)
    noise = torch.rand(batch_size, n_samples, device=device) * 2.0 - 1.0

    noise_window = torch.hann_window(n_fft, device=device)
    noise_stft = torch.stft(
        noise,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=noise_window,
        center=True,
        return_complex=True
    )  # (batch, 513, 251)
    H = F.pad(H, (0, 1), mode='replicate')
    Y = noise_stft * H

    # Inverse STFT perfectly overlap-adds the frames back into 1D audio
    filtered_noise = torch.istft(
        Y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=noise_window,
        center=True,
        length=n_samples
    )

    return filtered_noise

def additive_synth(
    amplitudes: torch.Tensor,  # Amplitudes (batch, n_sinusoids, n_frames)
    frequencies: torch.Tensor,  # Angular frequencies (batch, n_sinusoids, n_frames)
    n_samples: int,  # Number of samples to generate (upsample from n_frames)
    sample_rate: int = 16000
):
    # upsample
    amplitudes=F.interpolate(amplitudes,n_samples,mode='linear',align_corners=True)
    frequencies=F.interpolate(frequencies,n_samples,mode='linear',align_corners=True)
    
    phase = torch.zeros(frequencies[...,:1].shape)
    # append this to the frequencies list 
    angular_frequencies = 2 * torch.pi * frequencies / sample_rate
    phase = torch.cumsum(angular_frequencies, dim=-1)
    # now using the relation phase(n)= initial phase + sum(frequencies)
    y = amplitudes * torch.sin(phase)
    y = torch.sum(y, dim=1)
    return y

def get_harmonic_frequencies(
    f0: torch.Tensor,  # Fundamental frequency (Hz) (batch, n_frames)
    n_harmonics: int,  # Number of harmonics to generate
):
    rows=[]
    # for each frequency, we need to add n_harmonics rows downwards
    for h in range(1,n_harmonics+1):
        row = f0*h
        rows.append(row)
    frequencies=torch.stack(rows,dim=1)
    
    return frequencies

def harmonic_synth(
    f0: torch.Tensor,  # Fundamental frequency (Hz) (batch, n_frames)
    harmonic_amps: torch.Tensor,  # Amplitudes of harmonics (batch, n_harmonics, n_frames)
    n_samples: int,  # Number of samples to generate (upsample from n_frames)
):
    # Create the harmonic frequency envelopes
    n_harmonics = harmonic_amps.shape[1]
    frequencies = get_harmonic_frequencies(f0, n_harmonics) 
    return additive_synth(harmonic_amps, frequencies, n_samples=n_samples)

