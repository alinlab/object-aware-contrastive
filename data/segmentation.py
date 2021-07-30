import os
import numpy as np
from scipy.io import loadmat
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class SegmentationDataset(Dataset):
    """Base class for ReDO style segmentation datasets"""

    def __init__(self, root, split, split_root=None, transform=None, target_transform=None, seed=42):
        super().__init__()
        self.root = root
        self.split = split
        self.split_root = split_root
        self.ids = self._split_images(split=split)
        np.random.RandomState(seed).shuffle(self.ids)

        self.transform = transform
        self.target_transform = target_transform

    def _split_images(self, split):
        raise NotImplementedError

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        img = self._load_img(item)
        if self.transform is not None:
            img = self.transform(img)

        seg = self._load_seg(item)
        # assume fixed geometric transformation
        if self.target_transform is not None:
            seg = self.target_transform(seg)
            seg = self._post_process(seg)

        return img, seg

    def _load_img(self, item):
        raise NotImplementedError

    def _load_seg(self, item):
        raise NotImplementedError

    def _post_process(self, seg):
        seg = (seg > 0.5).float()  # binary mask
        return seg


class CUBSegmentation(SegmentationDataset):
    """CUB segmentation dataset, following ReDO splits"""

    def _split_images(self, split):
        split_id = {'train': 0, 'val': 1, 'test': 2}[split]
        splits = np.loadtxt(os.path.join(self.split_root, "cub.txt"), int)
        ids = np.loadtxt(os.path.join(self.root, "CUB_200_2011", "images.txt"), str)[:, 1]
        ids = ids[splits[:, 1] == split_id]
        return ids

    def _load_img(self, item):
        img = Image.open(os.path.join(self.root, "CUB_200_2011", "images",
                                      self.ids[item])).convert('RGB')
        return img

    def _load_seg(self, item):
        seg = Image.open(os.path.join(self.root, "CUB_200_2011", "segmentations",
                                      self.ids[item].replace('jpg', 'png'))).convert('L')
        return seg


class FlowersSegmentation(SegmentationDataset):
    """Flowers segmentation dataset, following ReDO splits"""

    def _split_images(self, split):
        split_id = {'train': 'tstid', 'val': 'valid', 'test': 'trnid'}[split]
        ids = loadmat(os.path.join(self.root, "flowers102", "setid.mat")).get(split_id)[0]
        ids = list(map('{:05d}'.format, ids))  # convert to string
        return ids

    def _load_img(self, item):
        imgname = "image_{}.jpg".format(self.ids[item])
        img = Image.open(os.path.join(self.root, "flowers102", "jpg", imgname)).convert('RGB')
        return img

    def _load_seg(self, item):
        segname = "segmim_{}.jpg".format(self.ids[item])
        seg = np.array(Image.open(os.path.join(self.root, "flowers102", "segmim", segname)))
        seg = 1 - ((seg[:, :, 0] == 0) + (seg[:, :, 1] == 0) + (seg[:, :, 2] == 254))
        seg = Image.fromarray((seg * 255).astype('uint8')).convert('L')
        return seg


class COCOSegmentation(SegmentationDataset):
    """COCO segmentation dataset"""

    def _split_images(self, split):
        from pycocotools.coco import COCO
        self.coco = COCO(os.path.join(self.root, 'COCO/annotations/instances_{}2017.json'.format(split)))
        ids = list(sorted(self.coco.imgs.keys()))
        return ids

    def _load_img(self, item):
        img_id = self.ids[item]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, 'COCO/{}2017'.format(self.split), path)).convert('RGB')
        return img

    def _load_seg(self, item, min_obj_scale=0):
        img_id = self.ids[item]
        img_size = (self.coco.imgs[img_id]['width'], self.coco.imgs[img_id]['height'])
        min_obj_size = img_size[0] * img_size[1] * min_obj_scale

        seg = np.zeros([img_size[1], img_size[0]])  # (H, W)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            if ann['area'] > min_obj_size:
                seg += self.coco.annToMask(ann)

        seg = Image.fromarray((seg.clip(0, 1) * 255).astype('uint8')).convert('L')
        return seg


class IN9Segmentation(SegmentationDataset):
    """ImageNet9 segmentation dataset"""

    def _split_images(self, split):
        if split == 'train':
            img_dir = os.path.join(self.root, 'imagenet9/original/train')
        else:
            img_dir = os.path.join(self.root, 'imagenet9/bg_challenge/original/val')
        ids = [img[0] for img in ImageFolder(img_dir).imgs]
        return ids

    def _load_img(self, item):
        return Image.open(self.ids[item]).convert('RGB')

    def _load_seg(self, item):
        seg = Image.open(self.ids[item].replace('original', 'only_fg')).convert('RGB')
        return seg

    def _post_process(self, seg):
        seg[seg > 0] = 1
        seg = (seg.sum(dim=0, keepdim=True) == 3).float()
        return seg
