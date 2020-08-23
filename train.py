import torch
import torch.nn as nn
import torch.optim as optim
from model import MusicModel
from config import get_config
from dataloader import get_loader
from tqdm import tqdm

seed = 4
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)   

def train(config):
    model = MusicModel()
    model.to(config.device)
    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)
    criterion = nn.BCELoss()

    train_loader = get_loader(config.train_path,config.batch_size)
    train_loader = iter(train_loader)

    for epoch in range(1,1+config.n_epoch):
        desc = '  - (Training|epoch:'+str(epoch)+') -  '
        for (data,label) in tqdm(train_loader,desc=desc,ncols=100):
            result = model(data).squeeze(-1).squeeze(-1)
            label = label.float()
            loss = criterion(result,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model,config.save_path)
  
if __name__ == "__main__":
    config = get_config()
    train(config)