import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys

class Conv(nn.Module):
    def __init__(self, ins, outs, activation=F.relu):
        super(Conv,self).__init__()
        self.conv1 = nn.Conv2d(ins,outs,3,padding=1)
        self.conv2 = nn.Conv2d(outs,outs,3,padding=1)
        self.activation = activation
        self.norm = nn.BatchNorm2d(outs)

    def forward(self,x):
        out = self.activation(self.norm(self.conv1(x)))
        out = self.activation(self.norm(self.conv2(out)))
        return out

class Up(nn.Module):
    def __init__(self, ins, outs, activation=F.relu):
        super(Up,self).__init__()
        self.up = nn.ConvTranspose2d(ins,outs,2,stride=2)
        self.conv1 = nn.Conv2d(ins,outs,3,padding=1)
        self.conv2 = nn.Conv2d(outs,outs,3,padding=1)
        self.activation = activation
        self.norm = nn.BatchNorm2d(outs)

    def crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self,x,bridge):
        up = self.up(x)
        crop = self.crop(bridge,up.size()[2])
        out = torch.cat([up,crop],1)
        out = self.activation(self.norm(self.conv1(out)))
        out = self.activation(self.norm(self.conv2(out)))
        return out

class MulWeight(nn.Module):
    def __init__(self, weight):
        super(MulWeight, self).__init__()
        self.w0 = weight[0]
        self.w1 = weight[1]
        self.w2 = weight[2]

    def forward(self,x):
        s0 = x[:,0,:,:]
        s1 = x[:,1,:,:]
        s2 = x[:,2,:,:]
        s0 = torch.mul(s0,self.w0)
        s1 = torch.mul(s1,self.w1)
        s2 = torch.mul(s2,self.w2)
        out = torch.stack([s0,s1,s2],1)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.activation = F.relu
        self.conv_1_8 = Conv(1,8)
        self.conv_8_16 = Conv(8,16)
        self.conv_16_32 = Conv(16,32)
        self.conv_32_64 = Conv(32,64)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.up1 = Up(64,32)
        self.up2 = Up(32,16)
        self.up3 = Up(16,8)
        self.last = nn.Conv2d(8,3,1)
        self.weight = MulWeight([0.1,1,1])

    def forward(self, x):
        block1 = self.conv_1_8(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_8_16(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_16_32(pool2)
        pool3 = self.pool3(block3)

        bottom = self.conv_32_64(pool3)

        up1 = self.up1(bottom,block3)

        up2 = self.up2(up1,block2)

        up3 = self.up3(up2,block1)

        raw_score = self.last(up3)

        score = self.weight(raw_score)

        return F.softmax(score)

def main():

    images, nc_ratio = [], []

    #0~1の値でできた、(画像の枚数,1,360,360)の4次元配列を用意して、入力とする。
    images = np.array(images).reshape(-1,1,360,360).astype(np.float32)

    #モデルのパスを指定して、学習済みのパラメーターをロードする。
    #NNのコードが書いてある必要がある。
    net = torch.load(model_path)

    #画像が多くなると、全てはメモリに乗り切らないので分割する。
    out = net(Variable(torch.from_numpy(images)).cuda())

    #nc比を計算する。
    _, pred = torch.max(out,1) #(n,360,360)で要素は0,1,2の配列
    pred = pred.cpu()
    pred = pred.data.numpy()
    pred = pred.reshape(-1,360,360)
    for x in pred:
        c = len(np.where(x>=1)[0])
        n = len(np.where(x==2)[0])
        ncr = n / c
        nc_ratio.append(ncr)
'''

if __name__ == '__main__':
    main()

'''