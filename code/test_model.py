import torch
import numpy as np
import glob
import re
import os
from build_model_old import SepConvNet
from torch.utils.data import DataLoader
from data_loader import SourceSepTest
from post_processing import reconstruct
from train_model import TimeFreqMasking
from tqdm import tqdm

if __name__ == '__main__':
    inp_size = [513,862]
    t1=1
    f1=513#513
    t2=12
    f2=1
    N1=50
    N2=30
    NN=128
    alpha = 0.001
    beta = 0.01
    beta_vocals = 0.03
    batch_size = 1
    num_epochs = 30

    destination_path= '../AudioResults/'
    phase_path = '../Processed/Phases/'
    vocals_directory='../AudioResults/vocals'
    drums_directory='../AudioResults/drums'
    bass_directory='../AudioResults/bass'
    others_directory='../AudioResults/others'

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    if not os.path.exists(vocals_directory):
        os.makedirs(vocals_directory)
    if not os.path.exists(drums_directory):
        os.makedirs(drums_directory)
    if not os.path.exists(bass_directory):
        os.makedirs(bass_directory)
    if not os.path.exists(others_directory):
        os.makedirs(others_directory)


    net = SepConvNet(t1,f1,t2,f2,N1,N2,inp_size,NN)
    # net.load_state_dict(torch.load('Weights/Weights_200_3722932.6015625.pth')) #least score Weights so far
    net.load_state_dict(torch.load('Weights/Weights_250_1268044.8148148148.pth'))
    net.eval()
    test_set = SourceSepTest(transforms = None)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False)
    for i,(test_inp,test_phase_file,file_str) in tqdm(enumerate(test_loader)):
        print('Testing, i='+str(i))
        test_phase = np.load(phase_path+test_phase_file[0])
        bass_mag, vocals_mag, drums_mag,others_mag = net(test_inp)
        bass_mag, vocals_mag, drums_mag,others_mag = TimeFreqMasking(bass_mag, vocals_mag, drums_mag,others_mag)
        regex = re.compile(r'\d+')
        index=regex.findall(file_str[0])
        reconstruct(test_phase, bass_mag, vocals_mag, drums_mag,others_mag,index[0],index[1],destination_path)

#    list = sorted(glob.glob('*.wav'))
