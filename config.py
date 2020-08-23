import argparse
import pprint
import torch


class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        # file_path
        self.chord_embedding_path = 'data/chord_embedding.pkl'
        self.music_path = 'data/music/'
        self.save_path = 'model/model.pkl'
        self.train_path = 'data/train.csv'
        self.test_path = 'data/test.csv'
        # device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Pre-trained Language Model
        self.PTM = 'chinese_wwm_ext_pytorch'

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config():
    parser = argparse.ArgumentParser()

    # # load setting
    # parser.add_argument('--checkpoint', type=str, default=None)
    
    # train setting
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--n_epoch', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-tb', '--test_batch_size', type=int, default=1)


    kwargs = parser.parse_args()
    
    # Namespace => Dictionary
    kwargs = vars(kwargs)

    return Config(**kwargs)


