import os
from tqdm import tqdm
from argparse import ArgumentParser

import torchvision.transforms.functional as TF
from torchvision.datasets import CocoDetection


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--min_size', type=float, default=0.01, help='minimum size (ratio) of cropped instance')
    parser.add_argument('--margin', type=float, default=0.2, help='expand bbox to contain some background')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, 'COCO')

    for split in ['train', 'val']:
        img_dir = os.path.join(data_dir, '{}2017'.format(split))
        ann_path = os.path.join(data_dir, 'annotations/instances_{}2017.json'.format(split))
        dataset = CocoDetection(img_dir, ann_path)

        save_dir = os.path.join(data_dir, 'objects', split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for img_idx, (img, anns) in enumerate(tqdm(dataset)):
            if len(anns) == 0:  # objects not exist
                continue

            min_size = img.size[0] * img.size[1] * args.min_size
            for ann_idx in range(len(anns)):
                if anns[ann_idx]['area'] < min_size:  # skip small objects
                    continue

                j, i, w, h = anns[ann_idx]['bbox']
                bbox = (i, j, h, w)  # convert to pytorch format
                bbox_expand = expand_bbox(*bbox, *img.size, margin=args.margin)
                img_cropped = TF.resized_crop(img, *bbox_expand, args.img_size)

                image_id = dataset.ids[img_idx]
                category_id = str(anns[ann_idx]['category_id'])
                if not os.path.exists(os.path.join(save_dir, category_id)):
                    os.mkdir(os.path.join(save_dir, category_id))

                save_path = os.path.join(save_dir, category_id, '{:012d}_{}.jpg'.format(image_id, ann_idx))
                img_cropped.convert('RGB').save(save_path)

    # create empty directories in training but not in validation dataset
    for category_id in os.listdir(os.path.join(data_dir, 'objects/train')):
        path = os.path.join(data_dir, 'objects/val', category_id)
        if not os.path.exists(path):
            os.makedirs(path)


def expand_bbox(i, j, h, w, W, H, margin=0.2):
    I, J = i + h, j + w
    i = max(0, i - h * margin)
    j = max(0, j - w * margin)
    I = min(H, I + h * margin)
    J = min(W, J + w * margin)
    h, w = I - i, J - j

    bbox = (i, j, h, w)
    return bbox


if __name__ == '__main__':
    cli_main()
