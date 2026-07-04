from src import dataset,features,models,synths
import torch
from torchaudio.functional import spectrogram
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

device= f'cuda:0' if torch.cuda.is_available() else 'cpu' 
def multiscale_spectral_loss(a_o,a_s) -> torch.Tensor:
    # spectrogram returns tensors of shape: 
    # (..., freq, time), freq is n_fft // 2 + 1 and n_fft is the number of Fourier bins, and time is the number of window hops (n_frame).
    fft_list=[2048,1024,512,256,128,64] # multiscale because of varying resolution
    total_loss=0.0
    for fft_size in fft_list:
        window=torch.hann_window(fft_size).to(a_o.device)
        S = spectrogram(a_o,pad=0,window=window,n_fft=fft_size,hop_length=fft_size//4,win_length=fft_size,power=1,normalized=False) # (32,1025,)
        S_hat = spectrogram(a_s,pad=0,window=window,n_fft=fft_size,hop_length=fft_size//4,win_length=fft_size,power=1,normalized=False)
        log_S = torch.log(S+1e-5)
        log_S_hat = torch.log(S_hat+1e-5)
        lin_loss=torch.sum(torch.abs(S-S_hat))
        log_loss=torch.sum(torch.abs(log_S-log_S_hat))
        total_loss+=lin_loss+log_loss
    
    return total_loss
    


def train(batch_size=32):
    data= dataset.DDSPDataset('/home/hasi/ddsp/data/f0_loudness_mfccs.pt')
    dl = DataLoader(data,batch_size,shuffle=True)
    # f0,loudness,mfccs= next(iter(dl))
    # print(f'f0 shape: {f0.shape} ')
    # print(f'loudness shape: {loudness.shape} ')
    # print(f'mfcc shape: {mfccs.shape} ')
    encoder = models.encoder(input_size=30).to(device)
    decoder=models.decoder().to(device)
    optimiser=Adam(list(encoder.parameters())+list(decoder.parameters()),lr=0.001)
    scheduler = ExponentialLR(optimiser, gamma=0.98)
    for epoch in tqdm(range(100)):
        for original_audio,f0,loudness,mfccs in dl:
            original_audio = original_audio.to(device)
            f0 = f0.to(device)
            loudness = loudness.to(device)
            mfccs = mfccs.to(device)
            optimiser.zero_grad()
            z=encoder(mfccs)
            audio_unnormalised,white_noise,f0_latent = decoder(f0=f0,z=z,l=loudness)
            f0_formatted=f0_latent.squeeze(-1)
            amps_formatted= audio_unnormalised.transpose(1, 2)
            harmonic_audio=synths.harmonic_synth(f0=f0_formatted,harmonic_amps=amps_formatted,n_samples=64000)
            noise_audio = synths.filtered_noise_synth(
                noise_magnitudes=white_noise, 
                n_samples=64000
            )
            synthesized_audio = harmonic_audio + noise_audio # shape: (batch_size,64000)
            loss=multiscale_spectral_loss(original_audio,synthesized_audio) 
            loss.backward()
            optimiser.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/100 Completed. Final Batch Loss: {loss.item():.4f}")
            
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': loss.item()
            }
            torch.save(checkpoint, f'checkpoints/ddsp_epoch_{epoch+1}.pt')
            print(f"--> Saved checkpoint to checkpoints/ddsp_epoch_{epoch+1}.pt")
        
    


train()