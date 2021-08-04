import torch
from torch.nn import functional as F

import random

from models.byol import BYOL


class BYOLMixup(BYOL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mixup(self, input, alpha):
        beta = torch.distributions.beta.Beta(alpha, alpha)
        randind = torch.randperm(input.shape[0], device=input.device)
        lam = beta.sample([input.shape[0]]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
        output = lam_expanded * input + (1. - lam_expanded) * input[randind]
        return output, randind, lam

    def training_step(self, batch, batch_idx):
        img1, img2 = self.process_batch(batch)

        img1, target_aux, lam = self.mixup(img1, alpha=1.)
        target = torch.arange(img1.shape[0], dtype=torch.long).cuda()

        self._momentum_update_key_encoder()

        # forward
        p1 = self.predictor(self.projector(self.encoder(img1)))
        p2 = self.predictor(self.projector(self.encoder(img2)))
        with torch.no_grad():
            y1 = self._projector(self._encoder(img1))
            y2 = self._projector(self._encoder(img2))

        # compute loss
        q = torch.cat([p1, y1])
        k = torch.cat([y2, p2])

        logits = q.mm(k.t())
        bsz = img1.size(0)

        x_ = torch.cat([torch.tensor(range(bsz)), torch.tensor(range(bsz))])
        y_ = torch.cat([target_aux, target_aux])
        lam_ = torch.cat([lam, lam])

        target_logits = lam_ * logits.diag() + (1. - lam_) * logits[x_, y_]
        loss = (2. - 2. * target_logits).mean()

        self.log_dict({'loss': loss}, prog_bar=True)

        return loss
