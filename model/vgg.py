import torch
import torch.nn as nn
import torchvision


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(32, 36):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cpu() * 2 - 1
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cpu() * 2

    def forward(self, X):  # relui_1
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5[:-2](h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cpu()
        self.criterion = nn.L1Loss()
        self.weights = [0.3, 0.3, 0.3, 0.3, 0.3]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

# #tes
# a=torch.randn(1,3,256,256)
# b=torch.randn(1,3,256,256)
# VG=VGGLoss()
# p=VG(a,b)
# for i in range(len(p)):
#     print(p[i].size());
