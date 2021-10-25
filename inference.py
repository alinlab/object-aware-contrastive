import os
from copy import deepcopy
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models import get_model_class, load_redo_model
from data import load_pretrain_datamodule, load_finetune_datamodule, load_segment_datamodule, get_image_ids
from data.transforms import get_normalization, FinetuneTransform
from utils import GradCAM, collect_outputs, accuracy
import utils.box_utils as box_utils


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    parser.add_argument('--ckpt_name', default=None, type=str, help='name of checkpoint directory')
    parser.add_argument("--ckpt_version", default=0, type=int, help="checkpoint version")
    parser.add_argument("--ckpt_epoch", default='last', type=str, help="checkpoint epoch")
    parser.add_argument("--seed", default=42, type=int, help="random seed")

    parser.add_argument("--mode", default='lineval', type=str, help="inference mode")
    parser.add_argument('--clf_type', default='lbfgs', type=str, help="classifier type")
    parser.add_argument('--save_path', default=None, type=str, help="save boxes/masks in the path")
    parser.add_argument('--cam_score', default='con', type=str, help="CAM score function")
    parser.add_argument('--expand_res', default=1, type=int, help="expand resolution for CAM")
    parser.add_argument('--cam_iters', default=1, type=int, help="CAM # of iterative refinements")
    parser.add_argument('--apply_crf', action='store_true', help="apply CRF for segmentation masks")
    parser.add_argument('--box_margin', default=0.2, type=float, help="margin for bounding boxes")
    parser.add_argument('--box_threshold', default=None, type=float, help="threshold for bounding boxes")
    parser.add_argument('--largest_box_only', action='store_true', help="store only the largest box")

    parser.add_argument("--model", default="moco", type=str, help="pre-training model")
    parser.add_argument("--normalize", default="coco", type=str, help="mean/std of pre-trained model")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")

    if parser.parse_known_args()[0].model != 'redo':
        model_class = get_model_class(parser.parse_known_args()[0].model)  # model class
        parser = model_class.add_model_specific_args(parser)  # model-specific arguments

    parser = pl.Trainer.add_argparse_args(parser)  # trainer arguments
    args = parser.parse_args()

    pl.seed_everything(1234)

    # load model
    if args.model in ['moco', 'byol']:
        # compatible with pretrained models
        import sys
        sys.path.insert(0, './data')
        
        args.ckpt_dir = os.path.join(args.log_dir, args.ckpt_name, 'version_{}'.format(args.ckpt_version))
        args.ckpt_path = os.path.join(args.ckpt_dir, 'checkpoints', '{}.ckpt'.format(args.ckpt_epoch))
        model = model_class.load_from_checkpoint(args.ckpt_path)

        if args.mode in ['seg', 'save_box', 'save_mask']:  # use CAM model
            model = GradCAM(model.encoder, projector=model.projector, expand_res=args.expand_res)

    elif args.model == 'redo':
        args.image_size = 128
        args.normalize = 'redo'
        model = load_redo_model(args.dataset)
    else:
        raise ValueError('No matching model class')

    print('Model: {}  Dataset: {}  Mode: {}'.format(args.model, args.dataset, args.mode))
    if args.mode == 'lineval':
        lineval(args, model)
    elif args.mode == 'seg':
        seg(args, model)
    elif args.mode == 'save_box':
        save_box(args, model)
    elif args.mode == 'save_mask':
        save_mask(args, model)
    else:
        raise ValueError('No matching inference mode')


def lineval(args, model, device='cuda'):
    model = model.to(device)
    model.eval()

    normalize = get_normalization(args.normalize)
    t_norm = FinetuneTransform(image_size=args.image_size, normalize=normalize, crop='center')

    dm = load_finetune_datamodule(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  train_transform=t_norm, test_transform=t_norm)

    print('Computing features...')
    with torch.no_grad():
        X_train, Y_train = collect_outputs(model, dm.train_dataloader())
        X_val, Y_val = collect_outputs(model, dm.val_dataloader())
        X_test, Y_test = collect_outputs(model, dm.test_dataloader())

    # train and evaluate linear classifier
    print('Evaluating classifier...')
    if args.clf_type == 'sgd':
        raise NotImplementedError

    elif args.clf_type == 'lbfgs':
        def build_step(X, Y, classifier, optimizer, weight):
            def step():
                optimizer.zero_grad()
                loss = F.cross_entropy(classifier(X), Y)
                for p in classifier.parameters():
                    loss = loss + p.pow(2).mean().mul(0.5 * weight)
                loss.backward()
                return loss
            return step

        X_train, Y_train = X_train.to(device), Y_train.to(device)
        X_val, Y_val = X_val.to(device), Y_val.to(device)
        X_test, Y_test = X_test.to(device), Y_test.to(device)

        classifier = nn.Linear(model.encoder.feat_dim, dm.num_classes).to(device)
        nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
        nn.init.normal_(classifier.bias, mean=0.0, std=0.01)
        optimizer = torch.optim.LBFGS(classifier.parameters(), max_iter=1000)

        best_acc = 0
        best_classifier = None
        for w in torch.logspace(-6, 5, steps=45).tolist():
            optimizer.step(build_step(X_train, Y_train, classifier, optimizer, w))
            train_acc = accuracy(X_train, Y_train, classifier)
            val_acc = accuracy(X_val, Y_val, classifier)
            test_acc = accuracy(X_test, Y_test, classifier)
            print('w: {:13.6f}  Train acc.: {:.2f}  Val acc.:{:.2f}  Test acc.:{:.2f}'.format(
                w, train_acc * 100, val_acc * 100, test_acc * 100))

            if val_acc > best_acc:
                best_acc = val_acc
                best_classifier = deepcopy(classifier)

        test_acc = accuracy(X_test, Y_test, best_classifier)
        print('Test acc.:{:.2f}'.format(test_acc * 100))
    else:
        raise NotImplementedError

    with open(os.path.join(args.ckpt_dir, 'lineval.txt'), 'a') as f:
        f.write('{}\t{}\te{}\t{:.2f}\n'.format(args.dataset, args.clf_type, args.ckpt_epoch, test_acc * 100))


def seg(args, model, device='cuda'):
    model = model.to(device)
    model.eval()

    normalize = get_normalization(args.normalize)
    t_norm = FinetuneTransform(image_size=args.image_size, normalize=normalize, crop='none')
    t_orig = FinetuneTransform(image_size=args.image_size, crop='none')

    print('Computing masks...')
    dm = load_segment_datamodule(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 transform=t_norm, target_transform=t_orig)
    loader = dm.test_dataloader()  # test loader

    pred_masks, gt_masks = compute_masks(args, model, loader)

    if args.apply_crf:
        loader.dataset.transform = t_orig   # original images
        images = torch.cat([x for x, _ in loader])
        pred_masks = box_utils.apply_crf(images, pred_masks)

    pred_masks = box_utils.clean_mask(pred_masks)

    print('Evaluating segmentation...')
    miou = box_utils.compute_mask_miou(gt_masks, pred_masks)
    print('Mask mIoU: {:.4f}'.format(miou))


def save_box(args, model, device='cuda'):
    model = model.to(device)
    model.eval()

    normalize = get_normalization(args.normalize)
    t_norm = FinetuneTransform(image_size=args.image_size, normalize=normalize, crop='none')
    t_orig = FinetuneTransform(image_size=args.image_size, crop='none')

    print('Computing masks...')
    dm = load_pretrain_datamodule(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  train_transform=t_norm, shuffle=False, drop_last=False)
    loader = dm.train_dataloader()  # train loader

    image_ids = get_image_ids(loader.dataset)
    pred_masks, _ = compute_masks(args, model, loader)

    if args.apply_crf:
        loader.dataset.transform = t_orig   # original images
        images = torch.cat([x for x, _ in loader])
        pred_masks = box_utils.apply_crf(images, pred_masks)

    pred_masks = box_utils.clean_mask(pred_masks, single=args.single_instance, min_obj_scale=0.01)
    pred_boxes = box_utils.extract_boxes(pred_masks, image_size=args.image_size, margin=args.box_margin)

    pred_boxes = {img_id: boxes for img_id, boxes in zip(image_ids, pred_boxes)}
    box_utils.save_boxes(pred_boxes, args.save_box_path)


def save_mask(args, model, device='cuda'):
    model = model.to(device)
    model.eval()

    normalize = get_normalization(args.normalize)
    t_norm = FinetuneTransform(image_size=args.image_size, normalize=normalize, crop='none')
    t_orig = FinetuneTransform(image_size=args.image_size, crop='none')

    print('Computing masks...')
    dm = load_pretrain_datamodule(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  train_transform=t_norm, shuffle=False, drop_last=False)
    loader = dm.train_dataloader()  # train loader

    image_ids = get_image_ids(loader.dataset)
    pred_masks, _ = compute_masks(args, model, loader)

    pred_masks = {img_id: mask for img_id, mask in zip(image_ids, pred_masks)}
    box_utils.save_masks(pred_masks, args.save_path, loader.dataset.root)


def compute_masks(args, model, loader):
    forward_kwargs = {}
    if isinstance(model, GradCAM):
        forward_kwargs['score_type'] = args.cam_score
        if args.cam_score == 'con':
            forward_kwargs['n_iters'] = args.cam_iters

    x, y = collect_outputs(model, loader, **forward_kwargs)
    return x, y


if __name__ == '__main__':
    cli_main()
