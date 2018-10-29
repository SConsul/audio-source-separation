import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from build_model import *
import os
from tqdm import tqdm
from data_loader import *
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#--------------------------
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    #property
    def avg(self):
        return self.sum / self.count
#------------------------------
# import csv
writer = SummaryWriter()
#----------------------------------------

inp_size = [513,3446]
t1=1
f1=513
t2=12
f2=1
N1=50
N2=30
NN=128
alpha = 0.001
beta = 0.01
beta_vocals = 0.03
batch_size = 30
num_epochs = 30

class MixedSquaredError(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MixedSquaredError, self).__init__()

    def forward(self, pred_bass,pred_vocals,pred_drums,pred_others, gt_bass,gt_vocals,gt_drums, gt_others):
        L_sq = np.sum(np.square(pred_bass-gt_bass)) + np.sum(np.square(pred_vocals-gt_vocals)) + np.sum(np.square(pred_drums-gt_drums))
        L_other = np.sum(np.square(pred_bass-gt_others)) + np.sum(np.square(pred_drums-gt_others)) + np.sum(np.square(pred_vocals-gt_others))
        L_othervocals = np.sum(np.square(pred_vocals - gt_others))
        L_diff = np.sum(np.square(pred_bass-pred_vocals)) + np.square(np.square(pred_bass-pred_drums)) + np.sum(np.square(pred_vocals-pred_drums))

        return L_sq - alpha*L_diff - beta*L_other - beta_vocals*L_othervocals

class TimeFreqMasking(nn.Module):
    def __init__(self):
        super(TimeFreqMasking, self).__init__()
    def forward(self,bass,vocals,drums,others):
        den = np.absolute(bass) + np.absolute(vocals) + np.absolute(drums) + np.absolute(others)
        bass = bass/den
        vocals = vocals/den
        drums = drums/den
        others = others/den

        return bass,vocals,drums,others

train_set = SourceSepTrain()

def train():
    cuda = torch.cuda.is_available()
    net = SepConvNet(t1,f1,t2,f2,N1,N2,inp_size,NN)
    criterion = MixedSquaredError()
    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print("preparing training data ...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")
    val_set = SourceSepVal()
    val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False)

    for epoch in range(num_epochs):
        train_loss = Average()

        net.train()
        for i, (inp, gt_bass,gt_drums,gt_vocals,gt_others) in tqdm(enumerate(train_loader)):
            inp = Variable(inp)
            gt_bass = Variable(gt_bass)
            gt_vocals = Variable(gt_vocals)
            gt_drums = Variable(gt_drums)
            gt_others= Variable(gt_others)
            if cuda:
                inp = inp.cuda()
                gt_bass = gt_bass.cuda()
                gt_vocals = gt_vocals.cuda()
                gt_drums = gt_drums.cuda()
                gt_others= gt_others.cuda()
            optimizer.zero_grad()
            o_bass, o_vocals, o_drums, o_others = net(inp)
            pred_bass,pred_vocals,pred_drums,pred_others = TimeFreqMasking(o_bass, o_vocals, o_drums, o_others)

            loss = criterion(pred_bass,pred_vocals,pred_drums,pred_others, gt_bass,gt_vocals,gt_drums,gt_others)
            writer.add_scalar('Train Loss',loss,epoch)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), images.size(0))
            for param_group in optimizer.param_groups:
                writer.add_scalar('Learning Rate',param_group['lr'])

        val_loss = Average()
        net.eval()
        for input, gt_bass,gt_drums,gt_vocals in tqdm(enumerate(train_loader)):
            input = Variable(input)
            gt_bass = Variable(gt_bass)
            gt_vocals = Variable(gt_vocals)
            gt_drums = Variable(gt_drums)
            if cuda:
                input = input.cuda()
                gt_vocals = gt_vocals.cuda()
                gt_drums = gt_drums.cuda()

            o_bass, o_vocals, o_drums, o_others = net(input)
            pred_bass,pred_vocals,pred_drums,pred_others = TimeFreqMasking(o_bass, o_vocals, o_drums, o_others)
            if (epoch)%10==0:
                writer.add_image('Validation Input',input,epoch)
                writer.add_image('Validation Bass GT ',gt_bass,epoch)
                writer.add_image('Validation Vocals GT ',gt_vocals,epoch)
                writer.add_image('Validation Drums GT ',gt_drums,epoch)

            vloss = criterion(pred_bass,pred_vocals,pred_drums,pred_others, gt_bass,gt_vocals,gt_drums)
            writer.add_scalar('Validation loss',vloss,epoch)
            val_loss.update(vloss.item(), images.size(0))

        print("Epoch {}, Training Loss: {}, Validation Loss: {}".format(epoch+1, train_loss.avg(), val_loss.avg()))
        torch.save(net.state_dict(), 'Weights_{}_{}.pth.tar'.format(epoch+1, val_loss.avg()))
    return net

def test(model):
    model.eval()


if __name__ == "__main__":
    train()
