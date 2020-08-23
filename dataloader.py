import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from config import get_config
from utils import get_MFCC,get_lyric_feature
import pickle
import math

config = get_config()


class MusicDataset(Dataset):
    def __init__(self, csv_file, max_len=300, dim=768):
        self.data = pd.read_csv(csv_file,encoding='utf-8',header=None)
        self.data = self.data.values.tolist()
        self.chord_embedding = torch.nn.Embedding(10,64)
        with open(config.chord_embedding_path,'rb') as f:
            pretrained_weight = pickle.load(f)
        self.chord_embedding.weight.data.copy_(pretrained_weight)
        self.C_to_N = {'A': 0, 'Am': 1, 'Bm': 2, 'C': 3, 'D': 4,'Dm': 5, 'E': 6, 'Em': 7, 'F': 8, 'G': 9}
        self.pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *-(math.log(10000.0) / dim)))
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.len = len(self.data)

    def __getitem__(self, idx):
        position = int(self.data[idx][1])
        lyric_feature = get_lyric_feature(self.data[idx][2])
        lyric_feature = lyric_feature + self.pe[position].unsqueeze(0)
        MFCC_feature = torch.tensor(get_MFCC(config.music_path+self.data[idx][0]+'.wav',self.data[idx][3],self.data[idx][4])).to(config.device)
        length = MFCC_feature.size()[0]
        if length > 1280:
            MFCC_feature = MFCC_feature[0:1280].to(config.device)
        if length < 1280:
            padding = torch.zeros(1280-length,13).to(config.device)
            MFCC_feature = torch.cat((MFCC_feature,padding),0).to(config.device)
    
        chord = self.data[idx][5]
        if chord!='_':
            chord = eval(chord)
            chord = [self.C_to_N[i] for i in chord]
        else:
            chord = []
        
        if len(chord)>20:
            chord = chord[0:20]
        lens = len(chord)
        if lens!=0:
            chord = torch.tensor(chord).to(config.device)
            chord_feature = self.chord_embedding(chord)
            chord_feature = chord_feature.view(lens*64,1)
        else:
            chord_feature = torch.tensor([[]])
            chord_feature = torch.transpose(chord_feature,1,0).to(config.device)
        length = chord_feature.size()[0]
        if length < 1280:
            padding = torch.zeros(1280-length,1).to(config.device)
            chord_feature = torch.cat((chord_feature,padding),0)
        res = [lyric_feature,MFCC_feature,chord_feature]
        return res, self.data[idx][6]

    def __len__(self):
        return self.len


def get_loader(csv_file,bs):
    dataset = MusicDataset(csv_file)
    dataloader = DataLoader(dataset=dataset, batch_size=bs, drop_last=True)
    return dataloader
