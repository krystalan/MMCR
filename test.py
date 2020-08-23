import torch
import torch.nn as nn
from model import MusicModel
from config import get_config
from dataloader import get_loader
from tqdm import tqdm

def test(model_path,test_path,bs):
    model = MusicModel()
    model = torch.load(model_path)

    test_loader = get_loader(test_path,bs)
    test_loader = iter(test_loader)

    TP = 0 
    FN = 0 
    FP = 0 
    TN = 0 

    desc = '  - (Testing) -  '
    for (data,label) in tqdm(test_loader,desc=desc,ncols=80):
        result = float(model(data).squeeze(-1).squeeze(-1))
        label = int(label[0])

        if label==1:
            if result >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if result >= 0.5:
                FP += 1
            else:
                TN += 1
    acc = float(TP+TN)/float(TP+FN+FP+TN)
    acc = round(acc*100,2)
    print('ACC:'+str(acc))


if __name__ == "__main__":
    config = get_config()
    test(config.save_path,config.test_path,config.test_batch_size)