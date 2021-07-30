from argparse import ArgumentParser
import math

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    """Base model for contrastive learning"""

    def __init__(
        self,
        arch: str = 'resnet18',
        image_size: int = 224,
        diff_transform: nn.Module = nn.Identity(),
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = load_backbone(arch, image_size)
        self.diff_transform = diff_transform

    def forward(self, x):
        return self.encoder(x)

    def process_batch(self, batch):
        inputs, target = batch
        inputs = [self.diff_transform(img) for img in inputs]
        return inputs

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):
        pass  # no validation error

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--arch", default="resnet18", type=str)
        parser.add_argument("--image_size", default=224, type=int)
        return parser


def load_backbone(arch, image_size):
    assert image_size in [32, 224]
    backbone = models.__dict__[arch](zero_init_residual=True)
    if image_size == 32:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    backbone.feat_dim = backbone.fc.weight.shape[1]
    backbone.fc = nn.Identity()
    reset_parameters(backbone)
    return backbone


def load_projection(n_in, n_hidden, n_out, num_layers=2, last_bn=False):
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    layers.append(Lambda(F.normalize))  # normalize projection
    projection = nn.Sequential(*layers)
    reset_parameters(projection)
    return projection


def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()
        if isinstance(m, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
