import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
import kornia.augmentation.functional as KF


def get_normalization(normalize):
    """Get normalization values for dataset"""
    if normalize == 'redo':
        return T.Normalize(mean=0.5, std=0.5)
    else:  # default: ImageNet
        return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class PretrainTransformBase:
    """Base transformations for pre-training"""
    def __init__(self, image_size=224, crop_scale=(0.08, 1.), view=2, use_mask=False):
        self.view = view
        transform = [
            T.RandomResizedCrop(image_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=0.5),
        ]
        transform_orig = [
            T.Resize((image_size, image_size)),
        ]

        if not use_mask:
            transform.append(T.ToTensor())
            transform_orig.append(T.ToTensor())
        self.transform = T.Compose(transform)
        self.transform_orig = T.Compose(transform_orig)


class PretrainTransform(PretrainTransformBase):
    """Base transformations for pre-training"""
    def __call__(self, sample, mask=None):
        if mask is None:
            out = []
            for _ in range(self.view):
                out.append(self.transform(sample))
        else:
            sample = T.ToTensor()(sample)
            mask = T.ToTensor()(mask)
            out = []
            for i in range(self.view):
                if i == 0:
                    combined = self.transform(torch.cat([sample, mask]))
                    first_sample = combined[:3, :, :]
                    mask = combined[-1, :, :]
                    out.append(first_sample)
                else:
                    out.append(self.transform(sample))
            out.append(mask)

        return out


class FinetuneTransform:
    """Base transformations for fine-tuning"""
    def __init__(self, image_size=224, crop='center', normalize=None):
        assert crop in ['random', 'center', 'none']

        transforms = []

        if crop == 'random':
            transforms += [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        elif crop == 'center':
            transforms += [
                T.Resize(image_size) if image_size != 224 else T.Resize(256),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        else:
            transforms += [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]

        if normalize is not None:
            transforms.append(normalize)

        self.transform = T.Compose(transforms)

    def __call__(self, sample):
        return self.transform(sample)


class KorniaTransform(nn.Module):
    """Other transformations on Kornia"""
    def __init__(self, image_size=224, normalize=None, jitter_strength=1., gaussian_blur=True):
        super().__init__()

        transforms = [
            ColorJitter(0.8 * jitter_strength,
                        0.8 * jitter_strength,
                        0.8 * jitter_strength,
                        0.2 * jitter_strength, p=0.8),
            K.RandomGrayscale(p=0.2),
        ]

        if gaussian_blur:
            kernel_size = int(0.1 * image_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            transforms.append(GaussianBlur(kernel_size, (0.1, 2.0)))

        if normalize is not None:
            normalize = self.to_kornia_normalize(normalize)
            transforms.append(normalize)

        self.transform = nn.Sequential(*transforms)

    def to_kornia_normalize(self, normalization):
        mean = torch.tensor(normalization.mean)
        std = torch.tensor(normalization.std)
        return K.Normalize(mean, std)

    def forward(self, x):
        return self.transform(x)


def apply_adjust_brightness(img1, params):
    ratio = params['brightness_factor'][:, None, None, None].to(img1.device)
    img2 = torch.zeros_like(img1)
    return (ratio * img1 + (1.0-ratio) * img2).clamp(0, 1)


def apply_adjust_contrast(img1, params):
    ratio = params['contrast_factor'][:, None, None, None].to(img1.device)
    img2 = 0.2989 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
    img2 = torch.mean(img2, dim=(-2, -1), keepdim=True)
    return (ratio * img1 + (1.0-ratio) * img2).clamp(0, 1)


class ColorJitter(K.ColorJitter):
    def apply_transform(self, x, params):
        transforms = [
            lambda img: apply_adjust_brightness(img, params),
            lambda img: apply_adjust_contrast(img, params),
            lambda img: KF.apply_adjust_saturation(img, params),
            lambda img: KF.apply_adjust_hue(img, params)
        ]

        for idx in params['order'].tolist():
            t = transforms[idx]
            x = t(x)

        return x


class GaussianBlur(K.AugmentationBase2D):
    def __init__(self, kernel_size, sigma, border_type='reflect',
                 return_transform=False, same_on_batch=False, p=0.5):
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def __repr__(self):
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape):
        return dict(sigma=torch.zeros(batch_shape[0]).uniform_(self.sigma[0], self.sigma[1]))

    def apply_transform(self, input, params):
        dtype = input.dtype  # function may change the original dtype
        sigma = params['sigma'].to(input.device)
        k_half = self.kernel_size // 2
        x = torch.linspace(-k_half, k_half, steps=self.kernel_size, dtype=input.dtype, device=input.device)
        pdf = torch.exp(-0.5 * (x[None, :] / sigma[:, None]).pow(2))
        kernel1d = pdf / pdf.sum(1, keepdim=True)
        kernel2d = torch.bmm(kernel1d[:, :, None], kernel1d[:, None, :])
        input = F.pad(input, (k_half, k_half, k_half, k_half), mode=self.border_type)
        input = F.conv2d(input.transpose(0, 1), kernel2d[:, None], groups=input.shape[0]).transpose(0, 1)
        return input.type(dtype)
