import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class _downConv(nn.Module):
    def __init__(self, nIn=3, nf=128, spectralNorm=False):
        super(_downConv, self).__init__()
        self.mods = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(nIn, nf // 4, 7, bias=False)) if spectralNorm
                else nn.Conv2d(nIn, nf // 4, 7, bias=False),
            nn.InstanceNorm2d(nf // 4, affine=True),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(nf // 4, nf // 2, 3, 2, 1, bias=False)) if spectralNorm
                else nn.Conv2d(nf // 4, nf // 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(nf // 2, affine=True),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(nf // 2, nf, 3, 2, 1, bias=False)) if spectralNorm
                else nn.Conv2d(nf // 2, nf, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(nf, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mods(x)


class _resBloc(nn.Module):
    def __init__(self, nf=128, spectralNorm=False):
        super(_resBloc, self).__init__()
        self.blocs = nn.Sequential(
            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)) if spectralNorm
                else nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(nf, affine=True),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)) if spectralNorm
                else nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )
        self.activationF = nn.Sequential(
            nn.InstanceNorm2d(nf, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.activationF(self.blocs(x) + x)


class _upConv(nn.Module):
    def __init__(self, nOut=3, nf=128, spectralNorm=False):
        super(_upConv, self).__init__()
        self.mods = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(nf, nf // 2, 3, 1, 1, bias=False)) if spectralNorm
                else nn.Conv2d(nf, nf // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(nf // 2, affine=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(nf // 2, nf // 4, 3, 1, 1, bias=False)) if spectralNorm
                else nn.Conv2d(nf // 2, nf // 4, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(nf // 4, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(nf // 4, nOut, 7, bias=True)) if spectralNorm
                else nn.Conv2d(nf // 4, nOut, 7, bias=True),
        )

    def forward(self, x):
        return self.mods(x)


class ReDO(nn.Module):
    def __init__(self, sizex=128, nIn=3, nMasks=2, nRes=3, nf=64, temperature=1):
        super(ReDO, self).__init__()
        self.nMasks = nMasks
        sizex = sizex // 4
        self.cnn = nn.Sequential(*([_downConv(nIn, nf)] +
                                   [_resBloc(nf=nf) for i in range(nRes)]))
        self.psp = nn.ModuleList([nn.Sequential(nn.AvgPool2d(sizex),
                                                nn.Conv2d(nf, 1, 1),
                                                nn.Upsample(size=sizex, mode='bilinear')),
                                  nn.Sequential(nn.AvgPool2d(sizex // 2, sizex // 2),
                                                nn.Conv2d(nf, 1, 1),
                                                nn.Upsample(size=sizex, mode='bilinear')),
                                  nn.Sequential(nn.AvgPool2d(sizex // 3, sizex // 3),
                                                nn.Conv2d(nf, 1, 1),
                                                nn.Upsample(size=sizex, mode='bilinear')),
                                  nn.Sequential(nn.AvgPool2d(sizex // 6, sizex // 6),
                                                nn.Conv2d(nf, 1, 1),
                                                nn.Upsample(size=sizex, mode='bilinear'))])
        self.out = _upConv(1 if nMasks == 2 else nMasks, nf + 4)
        self.temperature = temperature

    def forward(self, x):
        f = self.cnn(x)
        m = self.out(torch.cat([f] + [pnet(f) for pnet in self.psp], 1))
        m = torch.sigmoid(m / self.temperature)
        return m
