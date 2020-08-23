import torch
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer
import pandas as pd
import torch
from config import get_config
import scipy.io.wavfile
from python_speech_features import mfcc

config = get_config()
tokenizer = BertTokenizer.from_pretrained(config.PTM)
model = BertModel.from_pretrained(config.PTM)
model.to(config.device)

def get_lyric_feature(lyric):
    input_id = tokenizer.encode(lyric)
    input_id = torch.tensor([input_id])
    input_id = input_id.to(config.device)
    _ , pooled_output = model(input_id)
    return pooled_output



# def get_MFCC(root,idx):
#     data = pd.read_csv(root,encoding='utf-8',header=None)
#     data = data.values.tolist()
#     return eval(data[idx][4])


def get_MFCC(root,start,end):
    fs, sig = scipy.io.wavfile.read(root)
    start = int(float(start)*fs)-1
    if end == '[end]':
        return mfcc(sig[start::],fs).tolist()
    else:
        end = int(float(end)*fs)
        return mfcc(sig[start:end],fs).tolist()