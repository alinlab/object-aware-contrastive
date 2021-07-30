from argparse import ArgumentParser
from copy import deepcopy
import math

import torch
from torch.nn import functional as F
from pl_bolts.metrics import precision_at_k

from models.base import BaseModel, load_projection


class MoCo(BaseModel):
    """MoCo pre-training model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        feat_dim = self.encoder.feat_dim
        proj_dim = self.hparams.proj_dim

        self.projector = load_projection(feat_dim, feat_dim, proj_dim, num_layers=2, last_bn=False)

        self._encoder = deepcopy(self.encoder)
        self._projector = deepcopy(self.projector)

        for p in list(self._encoder.parameters()) + list(self._projector.parameters()):
            p.requires_grad = False

        # create negative queue
        num_negatives = self.hparams.num_negatives
        queue = F.normalize(torch.randn(num_negatives, proj_dim))
        self.register_buffer("queue", queue)
        self.queue_ptr = 0

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
        q = self.projector(self.encoder(img1))
        with torch.no_grad():
            k = self._projector(self._encoder(img2))

        # compute loss
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # positive logits: Nx1
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone()])  # negative logits: NxK
        logits = torch.cat([l_pos, l_neg], dim=1) / self.hparams.temperature  # logits: Nx(1+K)
        labels = torch.zeros(logits.size(0)).type_as(logits).long()  # labels: positive key indicators
        loss = F.cross_entropy(logits, labels)
        acc1, acc5 = precision_at_k(logits, labels, top_k=(1, 5))

        self._dequeue_and_enqueue(k)

        self.log_dict({'loss': loss, 'acc1': acc1, 'acc5': acc5}, prog_bar=True)
        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        em = self.hparams.encoder_momentum
        for online, target in [(self.encoder, self._encoder), (self.projector, self._projector)]:
            for p1, p2 in zip(online.parameters(), target.parameters()):
                p2.data = p2.data * em + p1.data * (1 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.trainer.distributed_backend in ('ddp', 'ddp_spawn', 'ddp2'):
            keys = torch.cat(self.all_gather(keys).unbind())

        ptr = self.queue_ptr
        self.queue[ptr:ptr + keys.size(0)] = keys
        self.queue_ptr = (ptr + keys.size(0)) % self.hparams.num_negatives

    def configure_optimizers(self):
        lr = self.hparams.base_lr * self.hparams.global_batch_size / 256  # linear lr scaling rule
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training params
        parser.add_argument("--base_lr", default=0.03, type=float)
        parser.add_argument("--final_lr", default=1e-3, type=float)
        parser.add_argument('--global_batch_size', default=None, type=int)  # default: inference mode
        parser.add_argument('--num_negatives', default=65536, type=int)
        parser.add_argument('--encoder_momentum', default=0.999, type=float)
        parser.add_argument("--temperature", default=0.2, type=float)
        parser.add_argument("--proj_dim", default=128, type=int)

        # transform params
        parser.add_argument("--jitter_strength", default=0.5, type=float)
        parser.add_argument("--gaussian_blur", default=True, type=bool)
        parser.add_argument("--min_crop_scale", default=0.08, type=float)  # stronger augmentation
        parser.add_argument("--max_crop_scale", default=1., type=float)

        return parser
