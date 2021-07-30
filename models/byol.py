from argparse import ArgumentParser
from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from models.base import BaseModel, load_projection


class BYOL(BaseModel):
    """BYOL pre-training model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        feat_dim = self.encoder.feat_dim
        proj_dim = self.hparams.proj_dim
        hidden_dim = self.hparams.hidden_dim

        self.projector = load_projection(feat_dim, hidden_dim, proj_dim, num_layers=2, last_bn=False)
        self.predictor = load_projection(proj_dim, hidden_dim, proj_dim, num_layers=2, last_bn=False)

        self._encoder = deepcopy(self.encoder)
        self._projector = deepcopy(self.projector)

        for p in list(self._encoder.parameters()) + list(self._projector.parameters()):
            p.requires_grad = False

        if self.hparams.global_batch_size is not None:
            self.lr_schedule = self.init_lr_schedule()  # custom lr schedule

    def init_lr_schedule(self):
        base_lr = self.hparams.base_lr * self.hparams.global_batch_size / 256  # linear lr scaling rule
        final_lr = self.hparams.final_lr
        max_epochs = self.hparams.max_epochs

        lr_schedule = torch.tensor([
            final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t / max_epochs))
            for t in torch.arange(max_epochs)
        ])
        return lr_schedule

    def on_train_epoch_start(self):
        # manually update learning rates
        optimizer = self.optimizers()
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[self.current_epoch]

        self.log('lr', self.lr_schedule[self.current_epoch])

    def training_step(self, batch, batch_idx):
        img1, img2 = self.process_batch(batch)

        self._momentum_update_key_encoder()

        # forward
        p1 = self.predictor(self.projector(self.encoder(img1)))
        p2 = self.predictor(self.projector(self.encoder(img2)))
        with torch.no_grad():
            y1 = self._projector(self._encoder(img1))
            y2 = self._projector(self._encoder(img2))

        # compute loss
        loss1 = F.cosine_similarity(p1, y2.detach(), dim=-1).mean()
        loss2 = F.cosine_similarity(p2, y1.detach(), dim=-1).mean()
        loss = -(loss1 + loss2) * 2  # l2 loss

        self.log_dict({'loss': loss}, prog_bar=True)
        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        em = self.hparams.encoder_momentum
        for online, target in [(self.encoder, self._encoder), (self.projector, self._projector)]:
            for p1, p2 in zip(online.parameters(), target.parameters()):
                p2.data = p2.data * em + p1.data * (1 - em)

    def configure_optimizers(self):
        lr = self.hparams.base_lr * self.hparams.global_batch_size / 256  # linear lr scaling rule
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        # do not use LARS optimizer (follow MoCo LR scheme)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training params
        parser.add_argument("--base_lr", default=0.03, type=float)  # follow MoCo LR scheme
        parser.add_argument("--final_lr", default=1e-3, type=float)
        parser.add_argument('--global_batch_size', default=None, type=int)  # default: inference mode
        parser.add_argument('--encoder_momentum', default=0.996, type=float)
        parser.add_argument("--hidden_dim", default=4096, type=int)
        parser.add_argument("--proj_dim", default=256, type=int)

        # transform params
        parser.add_argument("--jitter_strength", default=0.5, type=float)
        parser.add_argument("--gaussian_blur", default=True, type=bool)
        parser.add_argument("--min_crop_scale", default=0.08, type=float)  # stronger augmentation
        parser.add_argument("--max_crop_scale", default=1., type=float)

        return parser
