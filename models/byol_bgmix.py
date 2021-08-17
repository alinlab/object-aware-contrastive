from argparse import ArgumentParser

import torch
from torch.nn import functional as F

import random

from models.byol import BYOL
from models.bg_mixup import BGMixupModule


class BYOLBGMix(BYOL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bg_mixup_module = BGMixupModule()

    def process_batch(self, batch):
        # batch is a collection of (img1, img2, foreground mask of img1)
        masks = batch[-1]
        inputs = batch[:-1]
        inputs = [self.diff_transform(img) for img in inputs]
        inputs.append(masks)
        return inputs

    def training_step(self, batch, batch_idx):
        img1, img2, masks1 = self.process_batch(batch)
        img1, bg_only_img1 = self.bg_mixup_module.generate_bg_mixed_img(
            img1, masks1, self.hparams.cam_thres, self.hparams.aug_prob)

        return self.training_step_after_process_batch(img1, img2)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = BYOL.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training params
        parser.add_argument("--aug_prob", default=0.4, type=float)
        parser.add_argument("--cam_thres", default=0.2, type=float)

        return parser
