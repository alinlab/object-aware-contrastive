from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.moco import MoCo
from models.cutmix import cutmix
from pl_bolts.metrics import precision_at_k


class MoCoCutMix(MoCo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_step_after_process_batch(self, img1, img2):
        criterion = nn.CrossEntropyLoss(reduction='none')

        img1, target_aux, lam = cutmix(img1, alpha=self.hparams.alpha)
        target = torch.arange(img1.shape[0], dtype=torch.long).cuda()

        self._momentum_update_key_encoder()

        # forward
        q = self.projector(self.encoder(img1))
        with torch.no_grad():
            k = self._projector(self._encoder(img2))

        # compute loss
        contrast = torch.cat([k, self.queue.clone().detach()], dim=0)
        logits = torch.mm(q, contrast.t()) / self.hparams.temperature

        loss = (lam * criterion(logits, target) + (1. - lam) * criterion(logits, target_aux)).mean()
        acc1, acc5 = precision_at_k(logits, target, top_k=(1, 5))

        self._dequeue_and_enqueue(k)

        self.log_dict({'loss': loss, 'acc1': acc1, 'acc5': acc5}, prog_bar=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MoCo.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training params
        parser.add_argument("--alpha", default=1., type=float)
        return parser
