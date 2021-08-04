import torch
import torch.nn as nn
from torch.nn import functional as F

import random

from models.moco import MoCo
from pl_bolts.metrics import precision_at_k


class MoCoCutMix(MoCo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def unnormalize(self, data):
        if data.dim() == 4:
            data[:, 0, :, :] = 0.229 * data[:, 0, :, :] + 0.485
            data[:, 1, :, :] = 0.224 * data[:, 1, :, :] + 0.456
            data[:, 2, :, :] = 0.225 * data[:, 2, :, :] + 0.406
        else:
            data[0, :, :] = 0.229 * data[0, :, :] + 0.485
            data[1, :, :] = 0.224 * data[1, :, :] + 0.456
            data[2, :, :] = 0.225 * data[2, :, :] + 0.406

        return data

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
        criterion = nn.CrossEntropyLoss(reduction='none')

        img1, img2 = self.process_batch(batch)
        img1, target_aux, lam = self.cutmix(img1, alpha=1.)

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
