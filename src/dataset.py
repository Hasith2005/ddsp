import torch
from torch.utils.data import Dataset

class DDSPDataset(Dataset):
    def __init__(self,data_pt) -> None:
        super().__init__()
        self.data_pt=torch.load(data_pt)
        self.f0_t = self.data_pt['f0']
        self.loudness_t = self.data_pt['loudness']
        self.mfcc_t = self.data_pt['mfccs']
    def __len__(self):
        return self.data_pt['f0'].shape[0]
    def __getitem__(self, index):
        # return a single triple containig the f0,loudness and the MFCC thing given the index
        return (self.f0_t[index],self.loudness_t[index],self.mfcc_t[index])
