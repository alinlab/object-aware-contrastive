import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torchvision.ops.boxes import box_iou


def upscale_box(box, image_size):
    """Convert box to img_size scale"""
    x1, y1, x2, y2 = box
    W, H = get_image_size(image_size)
    box = (x1 * W, y1 * H, x2 * W, y2 * H)
    return box


def downscale_box(box, image_size):
    """Convert box to img_size scale"""
    x1, y1, x2, y2 = box
    W, H = get_image_size(image_size)
    box = (x1 / W, y1 / H, x2 / W, y2 / H)
    return box


def expand_box(box, image_size=(1, 1), margin=0.2):
    """Expand box with margin and downscale"""
    x1, y1, x2, y2 = box
    W, H = get_image_size(image_size)

    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * margin) / W
    y1 = max(0, y1 - h * margin) / H
    x2 = min(W, x2 + w * margin) / W
    y2 = min(H, y2 + h * margin) / H

    box = (x1, y1, x2, y2)
    return box


def get_image_size(image_size):
    """Convert image_size to (H,W) format"""
    if isinstance(image_size, (list, tuple)):
        W, H = image_size
    else:
        W = H = image_size
    return W, H


def xywh_to_xyxy(box):
    """Convert box from xywh to xyxy format"""
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    box = (x1, y1, x2, y2)
    return box


def xyxy_to_xywh(box):
    """Convert box from xyxy to xywh format"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    box = (x1, y1, w, h)
    return box


def extract_boxes(masks, image_size=224, threshold=0.5, margin=0, largest_only=False):
    """Extract all boxes from masks"""
    all_boxes = []
    for mask in masks:
        boxes = _extract_box(mask, threshold=threshold)

        if largest_only:
            areas = [box[2] * box[3] for box in boxes]
            boxes = [boxes[np.argmax(areas)]]

        for i, box in enumerate(boxes):
            box = xywh_to_xyxy(box)
            box = expand_box(box, image_size, margin=margin)
            boxes[i] = box

        all_boxes.append(boxes)

    return all_boxes


def _extract_box(mask, min_obj_scale=0.01, threshold=0.5, quantile=1.):
    """Extract boxes from mask"""
    _, h, w = mask.size()
    mask = mask.detach().cpu()

    threshold *= mask.view(1, -1).quantile(q=quantile, dim=1)

    mask = (mask > threshold).float()
    mask = np.array(T.ToPILImage()(mask))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        box = cv2.boundingRect(cnt)
        min_obj_size = h * w * min_obj_scale
        if box[2] * box[3] > min_obj_size:
            boxes.append(box)
    return boxes


def apply_crf(images, masks, n_iters=5):
    """Apply CRF post-processing"""
    import pydensecrf.densecrf as dcrf
    b, c, h, w = images.size()
    images = images.detach().cpu()
    masks = masks.detach().cpu()

    outs = []
    for i in tqdm(range(b), desc='Applying CRF...'):
        image = np.array(T.ToPILImage()(images[i]))
        mask = np.array(masks[i])
        mask = np.concatenate([mask, 1 - mask], axis=0).reshape((2, -1))

        d = dcrf.DenseCRF2D(h, w, 2)
        d.setUnaryEnergy(-np.log(mask + 1e-20))
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

        Q = d.inference(n_iters)
        out = np.array(Q).reshape((2, h, w))
        out = torch.tensor(out[0]).unsqueeze(0)
        outs.append(out)

    return torch.stack(outs, dim=0)


def clean_mask(masks, single=False, min_obj_scale=0.001):
    """Clean small noisy segmentations"""
    b, c, h, w = masks.size()
    masks = masks.detach().cpu()

    outs = []
    for mask in tqdm(masks, desc='Cleaning masks...'):
        mask = np.array(T.ToPILImage()(mask))
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

        if single:
            idxs = [stats[1:, -1].argmax() + 1] if len(stats) > 1 else []
        else:
            min_obj_size = h * w * min_obj_scale
            idxs = (stats[:, -1] > min_obj_size).nonzero()[0][1:]

        out = torch.zeros(output.shape)
        for idx in idxs:
            out[output == idx] = 1
        outs.append(out.view(1, h, w))

    return torch.stack(outs, dim=0)


def compute_box_ious(gt_boxes, pred_boxes):
    """Compute box mIoUs for all boxes"""
    ious = []
    for gt_box, pred_box in zip(gt_boxes, pred_boxes):
        iou = _compute_box_iou(gt_box, pred_box)
        if iou is not None:
            ious.append(iou)
    return ious


def _compute_box_iou(gt_box, pred_box):
    """Compute box mIoU"""
    if len(gt_box) > 0:
        if len(pred_box) > 0:
            gt_box = torch.tensor(gt_box)
            pred_box = torch.tensor(pred_box)
            ious, _ = box_iou(gt_box, pred_box).max(dim=1)  # best for each GT box
            iou = ious.mean().item()
        else:
            iou = 0
    else:
        iou = None
    return iou


def compute_mask_miou(gt_masks, pred_masks):
    """Compute mask mIoUs for all masks"""
    assert gt_masks.size() == pred_masks.size()
    gt_masks = (gt_masks > 0.5).float().flatten(start_dim=1)
    pred_masks = (pred_masks > 0.5).float().flatten(start_dim=1)

    inter = torch.min(gt_masks, pred_masks).sum(dim=1)
    union = torch.max(gt_masks, pred_masks).sum(dim=1)
    ious = inter / union
    ious[ious.isnan()] = 1  # both GT and pred are empty

    miou = ious.mean().item()
    return miou


def load_boxes(path):
    """Load boxes from path"""
    with open(path, "r") as f:
        lines = f.readlines()
    boxes = dict()
    for line in lines:
        toks = line.strip().split(" ")
        boxes[toks[0]] = []
        for tok in toks[1:]:
            box = list(map(float, tok.split(",")))
            boxes[toks[0]].append(box)
    return boxes


def save_boxes(d, path):
    """Save boxes from dictionary"""
    with open(path, "w") as f:
        for img_id, boxes in d.items():
            line = [img_id]
            for box in boxes:
                line.append(",".join(map("{:.6f}".format, box)))
            f.write(" ".join(line) + '\n')


def save_masks(d, path, root_path):
    """Save boxes from dictionary"""
    for img_id, mask in tqdm(d.items()):
        _, h, w = mask.size()

        original_img = np.asarray(Image.open(os.path.join(root_path, img_id)))
        h_original, w_original, _ = original_img.shape

        resized_mask = T.Resize((h_original, w_original))(mask)
        resized_mask = np.uint8(resized_mask.cpu() * 255)
        resized_mask = np.squeeze(resized_mask)
        resized_mask = Image.fromarray(resized_mask)

        save_path = os.path.join(path, img_id)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        resized_mask.save(save_path)
