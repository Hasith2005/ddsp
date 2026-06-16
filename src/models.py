import torch.nn as nn
import torch

device='cuda:0' if torch.cuda.is_available() else 'cpu'
class encoder(nn.Module):
    
    def __init__(self,input_size,hidden_size=512,num_layers=1,lin_output_size=16):
        super(encoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lin_output_size=lin_output_size
        self.ln = nn.LayerNorm(input_size)
        self.gru_encoder=nn.GRU(input_size,hidden_size,num_layers,batch_first=True) # output shape is going to be (batch_size,sequence_length,hidden_size)
        self.lin = nn.Linear(hidden_size,lin_output_size)
        
        
    def forward(self,z):
        # x: (batch_size,seq_length,hidden_size) this is going to be the MFCCs of shape (N,250,30)
        h0=torch.zeros(self.num_layers,z.size(0),self.hidden_size,device=z.device) # initial hidden state
        # pass the zs through a layernorm layer
        out=self.ln(z)
        out,_=self.gru_encoder(out,h0) # out.shape=(batch_size, seq_length, hidden_size) (N,250,512)
        out=self.lin(out)
        return out #shape (N,250,16)

class decoder(nn.Module):
    def return_linear_layers(self, input_size, hidden_size):
        return nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            )]
            +
            [nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU()
            ) for _ in range(2)]
        )
    def __init__(self,f0_dim=1,
                 z_dim=16,
                 l_dim=1,
                 mlp_hidden_size=512,
                 gru_input_size=512*3,
                 gru_hidden_size=512,
                 n_harmonics=101,
                 n_filter_mags=65): # f0 and l have shape [1,250,1]
        #all the MLPs have 3 layers and each layer has 512 units.
        super(decoder,self).__init__()
       
        self.lin_f0 = self.return_linear_layers(f0_dim,mlp_hidden_size)         
        self.lin_l =  self.return_linear_layers(l_dim,mlp_hidden_size)
        self.lin_z = self.return_linear_layers(z_dim,mlp_hidden_size)
        self.lin_fin = self.return_linear_layers(2*mlp_hidden_size+gru_hidden_size,mlp_hidden_size)
        self.gru = nn.GRU(gru_input_size,gru_hidden_size,batch_first=True)
        self.lin_harmonic_amps=nn.Linear(mlp_hidden_size,n_harmonics)
        self.lin_noise=nn.Linear(mlp_hidden_size,n_filter_mags)

    def mod_sigmoid(self,x):
        return 2*torch.pow(torch.sigmoid(x),2.3026)+1e-7
        
    def forward(self,f0:torch.Tensor,z:torch.Tensor,l:torch.Tensor):
        f0_latent=f0.clone()
        for block in self.lin_f0:
            f0=block(f0)
        for block in self.lin_l:
            l=block(l)
        for block in self.lin_z:
            z=block(z)            
        gru_in = torch.cat((f0,z,l),dim=-1) # so the shape is now (N,250,3*mlp_output_size)
        gru_out,h_n = self.gru(gru_in)
        # concat gru_out with f0 and l 
        mlp_in = torch.cat((f0,gru_out,l),dim=-1) # now the shape is (N,250,2*mlp_out_size+512)
        mlp_out = mlp_in
        for block in self.lin_fin:
            mlp_out=block(mlp_out) # now is of shape (N,250,mlp_hidden_size)
        audio_unnormalised = self.mod_sigmoid(self.lin_harmonic_amps(mlp_out))
        white_noise = self.mod_sigmoid(self.lin_noise(mlp_out))
        return audio_unnormalised,white_noise,f0_latent


        