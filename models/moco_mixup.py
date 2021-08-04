import torch
from torch.nn import functional as F

import random

from models.moco import MoCo
from pl_bolts.metrics import precision_at_k


class MoCoMixup(MoCo):
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
        q = self.projector(self.encoder(img1))
        with torch.no_grad():
            k = self._projector(self._encoder(img2))

        # compute loss
        contrast = torch.cat([k, self.queue.clone().detach()], dim=0)
        logits = torch.mm(q, contrast.t())
        loss = lam * F.cross_entropy(logits, target) + (1. - lam) * F.cross_entropy(logits, target_aux)
        acc1, acc5 = precision_at_k(logits, target, top_k=(1, 5))

        self._dequeue_and_enqueue(k)

        self.log_dict({'loss': loss, 'acc1': acc1, 'acc5': acc5}, prog_bar=True)
        return loss
