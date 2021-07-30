import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM(nn.Module):
    def __init__(self, encoder, classifier=None, projector=None, expand_res=1):
        super().__init__()
        self.encoder = self._expand_res(encoder, expand_res=expand_res)
        self.classifier = classifier
        self.projector = projector
        self.eval()

    def _expand_res(self, encoder, expand_res=1):
        """Expand resolution of penultimate feature"""
        assert expand_res in [1, 2, 4]

        def _reduce_stride(module):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    m.stride = (1, 1)

        if expand_res >= 2:
            _reduce_stride(encoder.layer4[0])
        if expand_res >= 4:
            _reduce_stride(encoder.layer3[0])

        return encoder

    def get_features(self, x):
        with torch.no_grad():
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
        return x

    def get_output(self, x):
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_projection(self, x):
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        return x

    def forward(self, image, score_type='con', **kwargs):
        if score_type == 'sup':
            cam = self.get_supervised_cam(image, **kwargs)
        elif score_type == 'con':
            cam = self.get_contrastive_cam(image, **kwargs)
        else:
            raise ValueError
        return cam

    def get_supervised_cam(self, image, label=None):
        b, c, h, w = image.size()

        feature = self.get_features(image)
        feature.requires_grad = True
        output = self.get_projection(feature)

        if label is None:
            label = output.max(dim=1)[1]

        score = output[range(output.size(0)), label]
        cam = self._compute_cam(feature, score, size=(h, w))

        return cam

    def get_contrastive_cam(self, image, n_iters=1, return_intermediate=False):
        b, c, h, w = image.size()
        image_original = image
        mask_color = image.mean(dim=(0, 2, 3), keepdim=True)

        key, queues = None, []
        _masks, _masked_images = [], []
        for it in range(n_iters):
            feature = self.get_features(image)
            feature.requires_grad = True
            output = self.get_projection(feature)

            if it == 0:
               key = output.detach()  # original images
            queues.append(output.detach())  # masked images

            score = self._contrastive_score(output, key, queues)
            cam = self._compute_cam(feature, score, size=(h, w), clamp_negative_weights=True)

            mask = torch.max(mask, cam) if it > 0 else cam  # union over iterations
            image = image_original * (1 - mask) + mask_color * mask

            _masks.append(cam)
            _masked_images.append(image)

        if return_intermediate:
            return mask, (_masks, _masked_images)
        else:
            return mask

    def _contrastive_score(self, query, key, queues):
        eye = torch.eye(query.size(0)).type_as(query)
        pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(0)
        neg = torch.cat([torch.einsum('nc,kc->nk', [query, queue]) * (1 - eye) for queue in queues], dim=1)
        score = (pos.exp().sum(dim=1) / neg.exp().sum(dim=1)).log()
        return score

    def _compute_cam(self, feature, score, size=(224, 224), clamp_negative_weights=True):
        grad = torch.autograd.grad(score.sum(), feature)[0]

        weight = F.adaptive_avg_pool2d(grad, output_size=(1, 1))
        if clamp_negative_weights:  # positive weights only
            weight = weight.clamp_min(0)

        cam = torch.sum(weight * feature, dim=1, keepdim=True).detach()

        cam = F.relu(cam)  # positive values only (follow Grad-CAM)
        cam = F.interpolate(cam, size=size, mode='bicubic', align_corners=False)
        cam = GradCAM.normalize(cam)

        return cam

    @staticmethod
    def normalize(cam, eps=1e-20):
        cam = cam.clone()
        for i in range(cam.size(0)):
            cam[i] -= torch.min(cam[i])
            cam[i] /= (torch.max(cam[i]) + eps)
        return cam

    @staticmethod
    def show_cam_on_image(images, masks):
        images = images.detach().cpu()
        masks = masks.detach().cpu()

        outs = []
        for image, mask in zip(images, masks):
            image = image.permute(1, 2, 0)
            mask = mask.squeeze(0)

            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[:, :, ::-1]  # BGR2RGB
            out = heatmap + np.float32(image)
            out = out / np.max(out)
            out = np.uint8(255 * out)

            out = torch.tensor(out).permute(2, 0, 1)
            outs.append(out)

        return torch.stack(outs, dim=0)

