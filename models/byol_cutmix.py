import torch
from torch.nn import functional as F

import random

from models.byol import BYOL


class BYOLCutMix(BYOL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cutmix(self, input, alpha):
        beta = torch.distributions.beta.Beta(alpha, alpha)
        randind = torch.randperm(input.shape[0], device=input.device)
        lam = beta.sample().to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        (bbx1, bby1, bbx2, bby2), lam = self.rand_bbox(input.shape[-2:], lam)
        output = input.clone()
        output[..., bbx1:bbx2, bby1:bby2] = output[randind][..., bbx1:bbx2, bby1:bby2]
        return output, randind, lam

    def rand_bbox(self, size, lam):
        W, H = size
        cut_rat = (1. - lam).sqrt()
        cut_w = (W * cut_rat).to(torch.long)
        cut_h = (H * cut_rat).to(torch.long)

        cx = torch.zeros_like(cut_w, dtype=cut_w.dtype).random_(0, W)
        cy = torch.zeros_like(cut_h, dtype=cut_h.dtype).random_(0, H)

        bbx1 = (cx - cut_w // 2).clamp(0, W)
        bby1 = (cy - cut_h // 2).clamp(0, H)
        bbx2 = (cx + cut_w // 2).clamp(0, W)
        bby2 = (cy + cut_h // 2).clamp(0, H)

        new_lam = 1. - (bbx2 - bbx1).to(lam.dtype) * (bby2 - bby1).to(lam.dtype) / (W * H)

        return (bbx1, bby1, bbx2, bby2), new_lam

    def training_step(self, batch, batch_idx):
        img1, img2 = self.process_batch(batch)
        img1, target_aux, lam = self.cutmix(img1, alpha=1.)

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

        target_logits = lam * logits.diag() + (1. - lam) * logits[x_, y_]
        loss = (2. - 2. * target_logits).mean()

        self.log_dict({'loss': loss}, prog_bar=True)

        return loss
