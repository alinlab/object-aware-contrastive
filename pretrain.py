from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models import get_model_class
from data import load_pretrain_datamodule
from data.transforms import get_normalization, PretrainTransform, FinetuneTransform, KorniaTransform
from utils import SSLOnlineEvaluator, ModelCheckpoint


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    parser.add_argument("--name", default=None, type=str, help="experiment name")
    parser.add_argument("--suffix", default=None, type=str, help="suffix for experiment name")
    parser.add_argument("--version", default=0, type=int, help="version (same as random seed)")
    parser.add_argument("--model", default="moco", type=str, help="pre-training model")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset")
    parser.add_argument("--ft_datasets", default=["coco"], nargs='*', type=str, help="datasets for fine-tuning")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--save_freq", default=25, type=int, help="save frequency of model")

    model_class = get_model_class(parser.parse_known_args()[0].model)  # model class

    parser = model_class.add_model_specific_args(parser)  # model-specific arguments
    parser = pl.Trainer.add_argparse_args(parser)  # trainer arguments
    args = parser.parse_args()

    pl.seed_everything(args.version)  # fix random seed

    # set default arguments
    if not isinstance(args.gpus, int):
        args.gpus = torch.cuda.device_count()
    args.benchmark = True

    args.check_val_every_n_epoch = 25
    args.save_freq = 25
    args.num_sanity_val_steps = 0

    args.global_batch_size = args.num_nodes * args.gpus * args.batch_size

    if args.name is None:  # default log name
        args.name = '_'.join([args.dataset, args.model, args.arch.replace('esnet', ''), f'b{args.global_batch_size}'])
    if args.suffix is not None:
        args.name += '_{}'.format(args.suffix)

    # define datamodule and model
    normalize = get_normalization(args.dataset)
    crop_scale = (args.min_crop_scale, args.max_crop_scale)

    train_transform = PretrainTransform(image_size=args.image_size, crop_scale=crop_scale, use_mask=('mask' in args.dataset))
    test_transform = FinetuneTransform(image_size=args.image_size, normalize=normalize)

    args.diff_transform = KorniaTransform(image_size=args.image_size, normalize=normalize,
                                          jitter_strength=args.jitter_strength,
                                          gaussian_blur=args.gaussian_blur)

    dm = load_pretrain_datamodule(args.dataset, ft_datasets=args.ft_datasets,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  train_transform=train_transform, test_transform=test_transform)

    model = model_class(**args.__dict__)

    # run experiments
    online_evaluator = SSLOnlineEvaluator(in_features=model.encoder.feat_dim, ft_datasets=args.ft_datasets)
    model_checkpoint = ModelCheckpoint(save_freq=args.save_freq)
    callbacks = [online_evaluator, model_checkpoint]

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger(args.log_dir, args.name, args.version),
        accelerator='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
