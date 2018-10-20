from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
import os
import numpy as np


class SourceSepTrain(Dataset):
    def __init__(self, path='data/train/'):
        self.path = path
        self.list = os.listdir(self.path)

    def __getitem__(self, index):
        mixture_path = 'data/train/Mixture/'
        base_path = 'data/train/Base/'
        vocals_path = 'data/train/Vocals/'
        drums_path = 'data/train/Drums/'
        others_path = 'data/train/Others/'

        mixture = torch.load(mixture_path+self.list[index]+'_m')
        phase = torch.load(mixture_path+self.list[index]+'_p')
        base = torch.load(base_path+self.list[index]+'_m')
        vocals = torch.load(vocals_path+self.list[index]+'_m')
        drums = torch.load(drums_path+self.list[index]+'_m')
        others = torch.load(others_path+self.list[index]+'_m')

        return (mixture, phase, base, vocals, drums, others)

    def __len__(self):
        return len(self.list)/6 # of how much data you have


class SourceSepTest(Dataset):
    def __init__(self, path='data/val/'):
        self.path = path
        self.list = os.listdir(self.path)

    def __getitem__(self, index):
        # stuff
        mixture_path = 'data/val/Mixture/'
        base_path = 'data/val/Base/'
        vocal_path = 'data/val/Vocals/'
        drums_path = 'data/val/Drums/'
        others_path = 'data/val/Others/'

        mixture = torch.load(mixture_path+self.list[index]+'_m')
        phase = torch.load(mixture_path+self.list[index]+'_p')
        base = torch.load(base_path+self.list[index]+'_m')
        vocals = torch.load(vocals_path+self.list[index]+'_m')
        drums = torch.load(drums_path+self.list[index]+'_m')
        others = torch.load(others_path+self.list[index]+'_m')

        return (mixture, phase, base, vocals, drums, others)

    def __len__(self):
        return len(self.list)/6
