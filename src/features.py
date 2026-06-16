# pitch and loudness extraction 
# will use the CREPE 
import torchcrepe
import numpy as np 
import torch
import torchaudio

def f_encoder(
    audio,
    sample_rate,
    step_size,
    fmin=50,
    fmax=550,
    model='tiny',
    batch_size=2048,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
):
    pitch = torchcrepe.predict(audio,
                           sample_rate,
                           step_size,
                           fmin,
                           fmax,
                           model,
                           batch_size=batch_size,
                           device=device)
    pitch=pitch[...,:250].unsqueeze(-1)
    
    return pitch

def a_weighting(freqs):
    f_sq = freqs**2
    ra = 12194**2*f_sq**2/((f_sq+20.6**2)*torch.sqrt((f_sq+107.7**2)*(f_sq+737.9**2))*(f_sq+12194**2))
    return 20*torch.log10(ra+1e-5)+2.0 # dB scale

    

def l_encoder(audio:torch.Tensor, # [1,16000]
                       frame_rate , # 100
                       sample_rate, # 16000 
                       ds_mean=None,
                       ds_std=None
                       ): 
    # replicate how the human ear would hear audio: we hear some frequencies louder than others. 
    # audio is of shape [1,16000]. we need to calculate the loudness every frame (100Hz). 
    hop_length = sample_rate//frame_rate
    n_fft=2048
    padded_audio = torch.nn.functional.pad(audio, (n_fft//2, n_fft//2), mode='reflect')
    audio_chunks= padded_audio.unfold(1,size=n_fft,step=hop_length) # overlapping chunks
    # do FFT for each chunk 
    window = torch.hann_window(n_fft, device=audio.device)
    windowed_chunks = audio_chunks * window
    X = torch.fft.rfft(windowed_chunks, dim=-1)
    power= X.abs()**2
    freqs = torch.fft.rfftfreq(n_fft, d=1/sample_rate).to(audio.device)
    a_db = a_weighting(freqs)
    # convert to linear scale so it can be a multiplier 
    a_lin = torch.pow(10,a_db/10)
    weighted_power = power * a_lin
    frame_loudness = weighted_power.sum(dim=-1)
    loudness_db=10 * torch.log10(frame_loudness+1e-5)
    if ds_mean is not None and ds_std is not None:
        loudness_db=(loudness_db-ds_mean)/ds_std
    loudness_db = loudness_db[:, :250].unsqueeze(-1)
    return loudness_db
    


if __name__=='__main__':
    # test: make a sinusoid 
    f0=250
    # sample_rate=16000
    duration =1
    # n_samples=duration*sample_rate
    # base = torch.ones((1,n_samples))*f0
    # angular_freq = 2*torch.pi*base/sample_rate
    # wave = torch.sin(torch.cumsum(angular_freq,dim=1))
    wave,sample_rate = torchaudio.load('data/nsynth-valid/audio/bass_electronic_018-042-100.wav')
    target_frame_rate = 250
    hop_length = sample_rate // target_frame_rate
    pitch_output = f_encoder(wave, sample_rate, hop_length)
    print(f"Pitch shape: {pitch_output.shape}")
    loudness_output = l_encoder(wave, target_frame_rate, sample_rate)
    print(f"Loudness shape: {loudness_output.shape}")