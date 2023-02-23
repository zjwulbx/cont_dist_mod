import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from torch.autograd import Function
from model.base_network import BaseNetwork, AdaptiveInstanceNorm
from model.Resnet import resnet34

# GradReverse.apply(inp, lambd)
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None


class Conv2dBlock(BaseNetwork):
    def __init__(self, fin, fout, kernel_size, padding, stride, param_free_norm_type='none', activation='relu'):
        super(Conv2dBlock, self).__init__()

        # create conv layers
        self.conv = spectral_norm(nn.Conv2d(fin, fout, kernel_size=kernel_size,
                                            padding=padding, stride=stride, padding_mode='reflect'))

        # define normalization layers
        if param_free_norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(fout)
        elif param_free_norm_type == 'batch':
            self.norm = nn.BatchNorm2d(fout)
        elif param_free_norm_type == 'adain':
            self.norm = AdaptiveInstanceNorm(128, fout)
        elif param_free_norm_type == 'none':
            self.norm = None
        else:
            raise ValueError('Unsupported norm %s' % param_free_norm_type)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Unsupported activation %s' % activation)

    def forward(self, x, s=None):
        x = self.conv(x)
        # for domain-level based style moodulation
        if isinstance(self.norm, AdaptiveInstanceNorm):
            x = self.norm(x, s)
        elif self.norm:
            x = self.norm(x)
            # for image-level based style modulation
            if s:
                mean, std = s
                x = x * std + mean
        if self.activation:
            x = self.activation(x)
        return x


class ContentEncoder(BaseNetwork):
    def __init__(self, nef=64):
        super(ContentEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            Conv2dBlock(3, nef, 3, 1, 1, 'instance', 'lrelu'),
            Conv2dBlock(nef, nef, 3, 1, 2, 'instance', 'lrelu')  # B*64*128*128
        )
        self.layer2 = nn.Sequential(
            Conv2dBlock(nef * 1, nef * 1, 3, 1, 1, 'instance', 'lrelu'),
            Conv2dBlock(nef * 1, nef * 2, 3, 1, 2, 'instance', 'lrelu'),  # B*128*64*64
        )
        self.layer3 = nn.Sequential(
            Conv2dBlock(nef * 2, nef * 2, 3, 1, 1, 'instance', 'lrelu'),
            Conv2dBlock(nef * 2, nef * 4, 3, 1, 2, 'instance', 'lrelu')  # B*256*32*32
        )
        self.layer4 = nn.Sequential(
            Conv2dBlock(nef * 4, nef * 4, 3, 1, 1, 'instance', 'lrelu'),
            Conv2dBlock(nef * 4, nef * 8, 3, 1, 2, 'instance', 'lrelu')  # B*512*16*16
        )
        self.layer5 = nn.Sequential(
            Conv2dBlock(nef * 8, nef * 8, 3, 1, 1, 'instance', 'lrelu'),
            Conv2dBlock(nef * 8, nef * 8, 3, 1, 2, 'instance', 'lrelu')  # B*512*8*8
        )
        self.layer6 = nn.Sequential(
            Conv2dBlock(nef * 8, nef * 8, 3, 1, 1, 'instance', 'lrelu'),
            Conv2dBlock(nef * 8, 1, 3, 1, 1, 'instance', 'lrelu')  # B*1*8*8
        )

    def forward(self, x, get_feature=False):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        c6 = self.layer6(c5)
        if get_feature:
            return [c6, c5, c4, c3, c2, c1]
        return c6



class Classifer(BaseNetwork):
    def __init__(self, ncls=10):
        super(Classifer, self).__init__()


        self.layer=resnet34(num_classes = ncls)

        self.fc3 = nn.Linear(ncls, ncls)
    def forward(self, x):

        return self.layer(x)


class AutoEncoder(BaseNetwork):
    def __init__(self, nf=64, ncls=292, noise=0.2):
        super(AutoEncoder, self).__init__()

        self.contentE = ContentEncoder(nf)
        self.C = Classifer(ncls)
        self.nf = nf
        self.embed = nn.Embedding(ncls, 128)
        self.noise = noise

    def forward(self, x, l, ratio=1.0):
        # z: content feature, scode: image-level style codes
        z, ccode = self.encode(x, ratio)
        # domain label to domain-level style codes
        out = self.classify(z)
        return out

    # introduce a classifier C with a gradient reversal layer to make the content feature domain-agnostic
    def classify(self, ccode):
        if isinstance(ccode, list):
            x = torch.cat(ccode, dim=1)
        else:
            x = ccode
        x = GradReverse.apply(x, 1)
        out = self.C(x)
        return out

    def encode(self, x, ratio=1.0):
        ccode = self.contentE(x)#B*1*8*8
        z = self.reparameterize(ccode)
        return z, ccode

    # add Gaussian noise of a fixed variance for robustness
    def reparameterize(self, ccode):
        return ccode + torch.randn_like(ccode) * self.noise