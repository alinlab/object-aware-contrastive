import os
import json
import random
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset

import utils.box_utils as box_utils


class COCO(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.join(data_root, 'COCO/{}2017'.format(split))
        self.coco = COCO(os.path.join(data_root, 'COCO/annotations/instances_{}2017.json'.format(split)))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        img_id = self.ids[item]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_id


class COCOBox(COCO):
    def __init__(self, data_root, split='train', box_path=None, transform=None):
        super().__init__(data_root, split=split, transform=transform)
        self.boxes = box_utils.load_boxes(box_path)

    def __getitem__(self, item):
        img_id = self.ids[item]
        boxes = self.boxes[str(img_id)]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if len(boxes) > 0:
            box = random.choice(boxes)
            box = box_utils.upscale_box(box, img.size)
            img = img.crop(box)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_id

    @staticmethod
    def get_gt_box(data_root, split='train', min_obj_scale=0.01, box_margin=0.2):
        from pycocotools.coco import COCO
        coco = COCO(os.path.join(data_root, 'COCO/annotations/instances_{}2017.json'.format(split)))
        ids = list(sorted(coco.imgs.keys()))

        all_boxes = {}
        for img_id in tqdm(ids):
            img_size = (coco.imgs[img_id]['width'], coco.imgs[img_id]['height'])
            min_obj_size = img_size[0] * img_size[1] * min_obj_scale

            boxes = []
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                if ann['area'] > min_obj_size:
                    box = box_utils.xywh_to_xyxy(ann['bbox'])
                    box = box_utils.expand_box(box, img_size, margin=box_margin)
                    boxes.append(box)

            all_boxes[str(img_id)] = boxes

        return all_boxes


class BaseDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


class Food101(BaseDataset):
    def __init__(self, data_root, split, transform=None):
        root = os.path.join(data_root, 'food-101')

        with open(os.path.join(root, 'meta', 'classes.txt')) as f:
            classes = [line.strip() for line in f]
        with open(os.path.join(root, 'meta', f'{split}.json')) as f:
            annotations = json.load(f)

        samples = []
        for i, cls in enumerate(classes):
            for path in annotations[cls]:
                samples.append((os.path.join(root, 'images', f'{path}.jpg'), i))

        super().__init__(samples, transform)


class Pets(BaseDataset):
    def __init__(self, data_root, split, transform=None):
        root = os.path.join(data_root, 'Pets')

        with open(os.path.join(root, 'annotations', f'{split}.txt')) as f:
            annotations = [line.split() for line in f]

        samples = []
        for sample in annotations:
            path = os.path.join(root, 'images', sample[0] + '.jpg')
            label = int(sample[1]) - 1
            samples.append((path, label))

        super().__init__(samples, transform)
