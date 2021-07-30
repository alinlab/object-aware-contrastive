import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as T
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule

from .datasets import *
from .segmentation import *

DATA_ROOT = '/data'
BOX_ROOT = './data/boxes'
MASK_ROOT = './data/masks'
SPLIT_ROOT = './data/splits'


def load_pretrain_datamodule(dataset, ft_datasets=(),
                             batch_size=64, num_workers=8, shuffle=True, drop_last=True,
                             train_transform=None, test_transform=None):

    if dataset in ['cub', 'flowers']:
        # segmentation datasets
        if dataset == 'cub':
            train = CUBSegmentation(DATA_ROOT, split_root=SPLIT_ROOT, split='train', transform=train_transform)
        else:
            train = FlowersSegmentation(DATA_ROOT, split='train', transform=train_transform)

    # COCO datasets
    elif dataset == 'coco':
        train = COCO(DATA_ROOT, split='train', transform=train_transform)

    elif 'coco-box' in dataset:
        box_path = os.path.join(BOX_ROOT, 'coco_{}.txt'.format(dataset[9:]))  # 'coco-box-{box_name}'
        train = COCOBox(DATA_ROOT, split='train', box_path=box_path, transform=train_transform)

    # IN9 datasets
    elif dataset == 'in9':
        data_dir = os.path.join(DATA_ROOT, 'bg_challenge/original/train')
        train = ImageFolder(data_dir, transform=train_transform)

    elif 'in9-mask' in dataset:
        #TODO: refactor
        data_dir = os.path.join(DATA_ROOT, 'bg_challenge/original/train')
        mask_dir = os.path.join(MASK_ROOT, 'in9_{}.txt'.format(dataset[9:]))  # 'in9-mask-{mask_name}'
        raise NotImplementedError('refactor code')

    else:
        raise NotImplementedError

    finetune = _load_online_finetune_datasets(ft_datasets, transform=test_transform)

    print('Pre-train dataset: {}'.format(len(train)))
    for i, name in enumerate(ft_datasets):
        print('FT-train dataset ({}): {}'.format(name, len(finetune[2 * i])))
        print('FT-test dataset ({}): {}'.format(name, len(finetune[2 * i + 1])))

    # create datamodule
    dm = BaseDataModule(
        train_dataset=train,
        val_dataset=finetune,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dm


def _load_online_finetune_datasets(ft_datasets=(), transform=None):
    """Load fine-tuning datasets for online evaluation"""
    if not isinstance(ft_datasets, (list, tuple)):
        ft_datasets = [ft_datasets]

    finetune = []
    for ft_dataset in ft_datasets:
        if ft_dataset == 'cifar10':
            # num_classes: 10
            ft_train = CIFAR10(DATA_ROOT, train=True, transform=transform)
            ft_test = CIFAR10(DATA_ROOT, train=False, transform=transform)
            finetune += [ft_train, ft_test]

        if ft_dataset == 'coco':
            # num_classes: 80
            ft_train = ImageFolder(os.path.join(DATA_ROOT, 'COCO/objects/train'), transform=transform)
            ft_test = ImageFolder(os.path.join(DATA_ROOT, 'COCO/objects/val'), transform=transform)
            ft_train = random_subset(ft_train, len(ft_train) // 5)  # use 20% of samples
            finetune += [ft_train, ft_test]

        if ft_dataset == 'in9':
            # num_classes: 9
            ft_train = ImageFolder(os.path.join(DATA_ROOT, 'bg_challenge/original/train'), transform=transform)
            ft_test = ImageFolder(os.path.join(DATA_ROOT, 'bg_challenge/original/val'), transform=transform)
            finetune += [ft_train, ft_test]

        if ft_dataset == 'cub':
            # num_classes: 200
            ft_train = ImageFolder(os.path.join(DATA_ROOT, 'CUB_200_2011/train'), transform=transform)
            ft_test = ImageFolder(os.path.join(DATA_ROOT, 'CUB_200_2011/test'), transform=transform)
            finetune += [ft_train, ft_test]

        if ft_dataset == 'food':
            # num_classes: 101
            ft_train = Food101(DATA_ROOT, split='train', transform=transform)
            ft_test = Food101(DATA_ROOT, split='test', transform=transform)
            finetune += [ft_train, ft_test]

        if ft_dataset == 'pets':
            # num_classes: 37
            ft_train = Pets(DATA_ROOT, split='trainval', transform=transform)
            ft_test = Pets(DATA_ROOT, split='test', transform=transform)
            finetune += [ft_train, ft_test]

        if ft_dataset == 'flowers':
            # num_classes: 102
            ft_train = ImageFolder(os.path.join(DATA_ROOT, 'flowers102/trn'), transform=transform)
            ft_test = ImageFolder(os.path.join(DATA_ROOT, 'flowers102/tst'), transform=transform)
            finetune += [ft_train, ft_test]

    return finetune


def load_finetune_datamodule(dataset, batch_size=64, num_workers=8,
                             train_transform=None, test_transform=None):
    loader_kwargs = {
        'train_transforms': train_transform, 'val_transforms': test_transform, 'test_transforms': test_transform,
        'batch_size': batch_size, 'num_workers': num_workers, 'drop_last': False, 'pin_memory': True,
    }

    generator = torch.Generator().manual_seed(42)

    def apply_all_transforms(train=None, val=None, test=None):
        apply_transform(train, train_transform)
        apply_transform(val, test_transform)
        apply_transform(test, test_transform)

    # create fine-tuning dataset
    if dataset == 'cifar10':
        dm = CIFAR10DataModule(DATA_ROOT, **loader_kwargs)

    elif dataset == 'cifar100':
        dm = CIFAR100DataModule(DATA_ROOT, **loader_kwargs)

    else:  # custom datamodules
        if dataset == 'cub':
            trainval = ImageFolder(os.path.join(DATA_ROOT, 'CUB_200_2011/train'))
            test = ImageFolder(os.path.join(DATA_ROOT, 'CUB_200_2011/test'))
            train, val = random_split(trainval, [4994, 1000], generator=generator)
            apply_all_transforms(train, val, test)
            num_classes = 200

        elif dataset == 'flowers':
            train = ImageFolder(os.path.join(DATA_ROOT, 'flowers102/trn'))
            val = ImageFolder(os.path.join(DATA_ROOT, 'flowers102/val'))
            test = ImageFolder(os.path.join(DATA_ROOT, 'flowers102/tst'))
            apply_all_transforms(train, val, test)
            num_classes = 102

        elif dataset == 'food':
            trainval = Food101(DATA_ROOT, split='train')
            test = Food101(DATA_ROOT, split='test')
            train, val = random_split(trainval, [68175, 7575], generator=generator)
            apply_all_transforms(train, val, test)
            num_classes = 101

        elif dataset == 'pets':
            trainval = Pets(DATA_ROOT, split='trainval')
            test = Pets(DATA_ROOT, split='test')
            train, val = random_split(trainval, [2940, 740], generator=generator)
            apply_all_transforms(train, val, test)
            num_classes = 37

        elif dataset == 'coco':
            trainval = ImageFolder(os.path.join(DATA_ROOT, 'COCO/objects/train'))
            test = ImageFolder(os.path.join(DATA_ROOT, 'COCO/objects/val'))
            train, val = random_split(trainval, [len(trainval) - 10000, 10000], generator=generator)
            apply_all_transforms(train, val, test)
            num_classes = 80

        else:
            raise NotImplementedError

        dm = BaseDataModule(
            train_dataset=train,
            val_dataset=val,
            test_dataset=test,
            num_classes=num_classes,
            **loader_kwargs,
        )

    dm.prepare_data()
    dm.setup()

    print('Train dataset: {}'.format(len(dm.train_dataloader().dataset)))
    print('Val dataset: {}'.format(len(dm.val_dataloader().dataset)))
    print('Test dataset: {}'.format(len(dm.test_dataloader().dataset)))

    return dm


def load_segment_datamodule(dataset, batch_size=64, num_workers=8,
                            transform=None, target_transform=None):

    dataset_kwargs = {'seed': 42, 'transform': transform, 'target_transform': target_transform}
    loader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'drop_last': False, 'pin_memory': True}

    if dataset == 'cub':
        test = CUBSegmentation(DATA_ROOT, split_root=SPLIT_ROOT, split='test', **dataset_kwargs)
    elif dataset == 'flowers':
        test = FlowersSegmentation(DATA_ROOT, split='test', **dataset_kwargs)
    elif dataset == 'coco':
        test = COCOSegmentation(DATA_ROOT, split='val', **dataset_kwargs)
    elif dataset == 'in9':
        test = IN9Segmentation(DATA_ROOT, split='test', **dataset_kwargs)
    else:
        raise ValueError

    dm = BaseDataModule(
        test_dataset=test,
        **loader_kwargs,
    )

    return dm


def get_image_ids(dataset):
    if isinstance(dataset, COCO):
        ids = map(str, dataset.ids)
    elif isinstance(dataset, SegmentationDataset):
        ids = dataset.ids
    elif isinstance(dataset, ImageFolder):
        ids = [path[0].replace(dataset.root + '/', '') for path in dataset.imgs]
    else:
        raise ValueError
    return ids


def random_subset(dataset, num_samples, generator=torch.default_generator):
    dataset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples], generator=generator)
    return dataset


def apply_transform(dataset, transform):
    if isinstance(dataset, Subset):
        apply_transform(dataset.dataset, transform)
    elif isinstance(dataset, Dataset):
        dataset.transform = transform


class BaseDataModule(pl.LightningDataModule):
    """Base datamodule"""

    def __init__(
            self,
            train_dataset=None,
            val_dataset=None,
            test_dataset=None,
            num_classes=None,
            num_workers: int = 16,
            batch_size: int = 32,
            pin_memory: bool = True,
            shuffle: bool = False,
            drop_last: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        if self.train_dataset is not None:
            self.num_samples = len(self.train_dataset)
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.train_dataset, shuffle=self.shuffle, drop_last=self.drop_last)

    def val_dataloader(self, *args, **kwargs):
        if isinstance(self.val_dataset, (list, tuple)):
            return [self._data_loader(dataset) for dataset in self.val_dataset]
        elif self.val_dataset is not None:
            return self._data_loader(self.val_dataset)
        else:
            return None

    def test_dataloader(self, *args, **kwargs):
        if isinstance(self.test_dataset, (list, tuple)):
            return [self._data_loader(dataset) for dataset in self.test_dataset]
        elif self.test_dataset is not None:
            return self._data_loader(self.test_dataset)
        else:
            return None

    def _data_loader(self, dataset, shuffle=False, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=self.pin_memory
        )


class CIFAR100DataModule(CIFAR10DataModule):
    """CIFAR100 datamodule"""
    from torchvision.datasets import CIFAR100

    name = "cifar100"
    dataset_cls = CIFAR100

    @property
    def num_classes(self) -> int:
        return 100
