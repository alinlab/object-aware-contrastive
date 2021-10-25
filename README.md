# Object-aware Contrastive Learning

Official PyTorch implementation of
[**"Object-aware Contrastive Learning for Debiased Scene Representation"**](https://arxiv.org/abs/2108.00049) (NeurIPS 2021) by
[Sangwoo Mo*](https://sites.google.com/view/sangwoomo),
[Hyunwoo Kang*](https://github.com/hyunOO),
[Kihyuk Sohn](https://sites.google.com/site/kihyuksml),
[Chun-Liang Li](https://chunliangli.github.io/),
and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html).

## Installation

Install required libraries.
```
pip install -r requirements.txt
```
Download datasets in `/data` (e.g., `/data/COCO`).

Pretrained models trained on COCO are available at [google drive](https://drive.google.com/drive/folders/1ii-j0fPnAmy4LdecdgvwsD3o4DljOjl4?usp=sharing).

## Train models

Logs will be saved in `logs/{dataset}_{model}_{arch}_b{global_batch_size}` directory,
  where `global_batch_size = num_nodes * gpus * batch_size` (default batch size = `64 * 4 = 256`).

### Step 1. Train vanilla models

Train vanilla models (change `dataset` and `ft_datasets` as `cub` or `in9`).
```
python pretrain.py --dataset coco --model moco --arch resnet18\
    --ft_datasets coco --batch_size 64 --max_epochs 800
```

### Step 2. Pre-compute CAM masks
Pre-compute bounding boxes for object-aware random crop.
```
python inference.py --mode save_box --model moco --arch resnet18\
    --ckpt_name coco_moco_r18_b256 --dataset coco\
    --expand_res 2 --cam_iters 10 --apply_crf\
    --save_path data/boxes/coco_cam-r18.txt
```
Pre-compute masks for background mixup.
```
python inference.py --mode save_mask --model moco --arch resnet18\
    --ckpt_name in9_moco_r18_256 --dataset in9\
    --expand_res 1 --cam_iters 1\
    --save_path data/masks/in9_cam-r18
```

### Step 3. Re-train debiased models
Train contextual debiased model with object-aware random crop.
```
python pretrain.py --dataset coco-box-cam-r18 --model moco --arch resnet18\
     --ft_datasets coco --batch_size 64 --max_epochs 800
```
Train background debiased model with background mixup.
```
python pretrain.py --dataset in9-mask-cam-r18 --model moco_bgmix --arch resnet18\
    --ft_datasets in9 --batch_size 64 --max_epochs 800
```


## Evaluate models

### Linear evaluation
```
python inference.py --mode lineval --model moco --arch resnet18\
    --ckpt_name coco_moco_r18_b256 --dataset coco
```

### Object localization
```
python inference.py --mode seg --model moco --arch resnet18\
    --ckpt_name cub200_moco_r18_b256 --dataset cub200\
    --expand_res 2 --cam_iters 10 --apply_crf
```

### Detection & Segmentation (fine-tuning)
```
mv detection
python convert-pretrain-to-detectron2.py coco_moco_r50.pth coco_moco_r50.pkl
python train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 8\
    MODEL.WEIGHTS weights/coco_moco_r18.pkl
```
