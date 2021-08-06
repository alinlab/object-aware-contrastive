from argparse import ArgumentParser

import torch
from torch.nn import functional as F

from models.byol import BYOL
from models.mixup import mixup


class BYOLMixup(BYOL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_step_after_process_batch(self, img1, img2):
        img1, target_aux, lam = mixup(img1, alpha=1.)
        target = torch.arange(img1.shape[0], dtype=torch.long).cuda()

        self._momentum_update_key_encoder()

        # forward
        p1 = self.predictor(self.projector(self.encoder(img1)))
        p2 = self.predictor(self.projector(self.encoder(img2)))
        with torch.no_grad():
            y1 = self._projector(self._encoder(img1))
            y2 = self._projector(self._encoder(img2))

        loss1 = (F.cosine_similarity(p1, y2.detach(), dim=-1) * lam + F.cosine_similarity(p1, y2[target_aux, :].detach(), dim=-1) * (1-lam)).mean()
        loss2 = (F.cosine_similarity(p2, y1.detach(), dim=-1) * lam + F.cosine_similarity(p2[target_aux, :], y1.detach(), dim=-1) * (1 -lam)).mean()
        loss = -(loss1 + loss2) * 2  # l2 loss

        self.log_dict({'loss': loss}, prog_bar=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = BYOL.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training params
        parser.add_argument("--alpha", default=1., type=float)
        return parser
