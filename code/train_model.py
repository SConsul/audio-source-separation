import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from build_model_old import *
from cyclicAnnealing import CyclicLinearLR
import os
from tqdm import tqdm
from data_loader import *
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
mean_var_path= "../Processed/"
if not os.path.exists('Weights'):
    os.makedirs('Weights')
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

inp_size = [513,52]
t1=1
f1=513#513
t2=12
f2=1
N1=50
N2=30
NN=128
alpha = 0.003
beta = 0.03
beta_vocals = 0.09
batch_size = 30
num_epochs = 100


class MixedSquaredError(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MixedSquaredError, self).__init__()

    def forward(self, pred_bass,pred_vocals,pred_drums,pred_others, gt_bass,gt_vocals,gt_drums, gt_others):


        L_sq = torch.sum((pred_bass-gt_bass).pow(2)) + torch.sum((pred_vocals-gt_vocals).pow(2)) + torch.sum((pred_drums-gt_drums).pow(2))
        L_other = 2*torch.sum((pred_bass-gt_others).pow(2)) + torch.sum((pred_drums-gt_others).pow(2))
        #+ torch.sum((pred_vocals-gt_others).pow(2))
        L_othervocals = torch.sum((pred_vocals - gt_others).pow(2))
        L_diff = torch.sum((pred_bass-pred_vocals).pow(2)) + torch.sum((pred_bass-pred_drums).pow(2)) + 15*torch.sum((pred_vocals-pred_drums).pow(2))

        return L_sq- alpha*L_diff - beta*L_other - beta_vocals*L_othervocals

def TimeFreqMasking(bass,vocals,drums,others):
    den = torch.abs(bass) + torch.abs(vocals) + torch.abs(drums) + torch.abs(others)
    bass = torch.abs(bass)/den
    vocals = torch.abs(vocals)/den
    drums = torch.abs(drums)/den
    others = torch.abs(others)/den
    return bass,vocals,drums,others
#mu=torch.load(os.path.join(mean_var_path,'mean.pt'))
#std=torch.load(os.path.join(mean_var_path,'std.pt'))
#transformations_train = transforms.Compose([transforms.Normalize(mean = mu, std = std)])
#
train_set = SourceSepTrain(transforms = None)


#transformation_test = transforms.Compose([ transforms.Normalize(mean = 0.0, std =1./var), transforms.Normalize(mean = -1*mu, std = 1.0),])


def train():
    cuda = torch.cuda.is_available()
    net = SepConvNet(t1,f1,t2,f2,N1,N2,inp_size,NN)
    criterion = MixedSquaredError()
    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    #scheduler = CyclicLinearLR(optimizer, milestones=[60,120])
    scheduler = MultiStepLR(optimizer, milestones=[60,120])
    print("preparing training data ...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")
    val_set = SourceSepVal(transforms = None)
    val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False)

    for epoch in range(num_epochs):
        scheduler.step()
        train_loss = Average()

        net.train()
        for i, (inp, gt_bass,gt_vocals,gt_drums,gt_others) in enumerate(train_loader):
            mean = torch.mean(inp)
            std = torch.std(inp)
            inp_n = (inp-mean)/std

            inp_n = Variable(inp)
            gt_bass = Variable(gt_bass)
            gt_vocals = Variable(gt_vocals)
            gt_drums = Variable(gt_drums)
            gt_others= Variable(gt_others)
            if cuda:
                inp_n = inp_n.cuda()
                gt_bass = gt_bass.cuda()
                gt_vocals = gt_vocals.cuda()
                gt_drums = gt_drums.cuda()
                gt_others= gt_others.cuda()
            optimizer.zero_grad()
            o_bass, o_vocals, o_drums, o_others = net(inp_n)


            mask_bass,mask_vocals,mask_drums,mask_others = TimeFreqMasking(o_bass, o_vocals, o_drums, o_others)
            pred_drums=inp*mask_drums
            pred_vocals=inp*mask_vocals
            pred_bass=inp*mask_bass
            pred_others=inp*mask_others

            loss = criterion(pred_bass,pred_vocals,pred_drums,pred_others, gt_bass,gt_vocals,gt_drums,gt_others)
            writer.add_scalar('Train Loss',loss,epoch)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), inp.size(0))
            for param_group in optimizer.param_groups:
                writer.add_scalar('Learning Rate',param_group['lr'])

        val_loss = Average()
        net.eval()
        for i,(val_inp, gt_bass,gt_vocals,gt_drums,gt_others) in enumerate(val_loader):
            #val_mean = torch.mean(val_inp)
            #val_std = torch.std(val_inp)
            #val_inp_n = (val_inp-val_mean)/val_std

            val_inp = Variable(val_inp)
            gt_bass = Variable(gt_bass)
            gt_vocals = Variable(gt_vocals)
            gt_drums = Variable(gt_drums)
            gt_others = Variable(gt_others)
            if cuda:
                val_inp_n = val_inp_n.cuda()
                gt_bass = gt_bass.cuda()
                gt_vocals = gt_vocals.cuda()
                gt_drums = gt_drums.cuda()
                gt_others = gt_others.cuda()

            o_bass, o_vocals, o_drums, o_others = net(val_inp)
            mask_bass,mask_vocals,mask_drums,mask_others = TimeFreqMasking(o_bass, o_vocals, o_drums, o_others)
            #print(val_inp.shape)
            #print(mask_drums.shape)
            #assert False
            pred_drums=val_inp*mask_drums
            pred_vocals=val_inp*mask_vocals
            pred_bass=val_inp*mask_bass
            pred_others=val_inp*mask_others

            if (epoch)%10==0:
                writer.add_image('Validation Input',val_inp,epoch)
                writer.add_image('Validation Bass GT ',gt_bass,epoch)
                writer.add_image('Validation Bass Pred ',pred_bass,epoch)
                writer.add_image('Validation Vocals GT ',gt_vocals,epoch)
                writer.add_image('Validation Vocals Pred ',pred_vocals,epoch)
                writer.add_image('Validation Drums GT ',gt_drums,epoch)
                writer.add_image('Validation Drums Pred ',pred_drums,epoch)
                writer.add_image('Validation Other GT ',gt_others,epoch)
                writer.add_image('Validation Others Pred ',pred_others,epoch)

            vloss = criterion(pred_bass,pred_vocals,pred_drums,pred_others, gt_bass,gt_vocals,gt_drums, gt_others)
            writer.add_scalar('Validation loss',vloss,epoch)
            val_loss.update(vloss.item(), inp.size(0))

        print("Epoch {}, Training Loss: {}, Validation Loss: {}".format(epoch+1, train_loss.avg(), val_loss.avg()))
        torch.save(net.state_dict(), 'Weights/Weights_{}_{}.pth'.format(epoch+1, val_loss.avg()))
    return net

def test(model):
    model.eval()


if __name__ == "__main__":
    train()
