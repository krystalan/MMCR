import torch
import torch.nn as nn
import pickle
import math
from config import get_config

config = get_config()

class MusicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actv = nn.ReLU()
        self.music_layer = nn.Linear(13,10)
        self.music_layer_2 = nn.Linear(10,1)
        self.classifier1 = nn.Linear(3328,1000)
        self.classifier2 = nn.Linear(1000,512)
        self.classifier3 = nn.Linear(512,1)

    def forward(self,model_input):
        lyric_feature = model_input[0]
        MFCC_feature = model_input[1]
        chord_feature = model_input[2]

        lyric_feature = torch.transpose(lyric_feature,2,1)

        music_feature = self.music_layer(MFCC_feature)
        music_feature = self.music_layer_2(music_feature)

        all_feature = torch.cat((lyric_feature,music_feature,chord_feature),1)
        all_feature = torch.transpose(all_feature,2,1)

        output = self.actv(self.classifier1(all_feature))
        output = self.actv(self.classifier2(output))
        output = self.classifier3(output)
        return torch.sigmoid(output)