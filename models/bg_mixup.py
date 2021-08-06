import torch
from torch.nn import functional as F

import random


class Queue():
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.memory = []

    def push(self, elem):
        self.memory.append(elem)
        while len(self.memory) > self.max_len:
            self.memory.pop(0)

    def get(self):
        return random.choice(self.memory)

    def __len__(self):
        return len(self.memory)


class BGMixupModule():
    def __init__(self):
        self.bg_queue = Queue()

    def get_minimum_rectangle_including_mask(self, mask):
        if mask.dim() == 4:
            non_zero = (mask > 0.1).nonzero()
            h_min = non_zero[:, 2].min()
            h_max = non_zero[:, 2].max()
            w_min = non_zero[:, 3].min()
            w_max = non_zero[:, 3].max()
        elif mask.dim() == 3:
            non_zero = (mask > 0.1).nonzero()
            h_min = non_zero[:, 1].min()
            h_max = non_zero[:, 1].max()
            w_min = non_zero[:, 2].min()
            w_max = non_zero[:, 2].max()
        elif mask.dim() == 2:
            non_zero = (mask > 0.1).nonzero()
            if non_zero.size(0) == 0 or non_zero.size(1) == 0:
                return 0, mask.size(0) - 1, 0, mask.size(1) - 1
            h_min = non_zero[:, 0].min()
            h_max = non_zero[:, 0].max()
            w_min = non_zero[:, 1].min()
            w_max = non_zero[:, 1].max()
        else:
            raise ValueError
        return h_min, h_max, w_min, w_max

    def generate_bg_only_img(self, img, mask, h_min, h_max, w_min, w_max, aug_prob):
        c, height, width = img.size()
        if h_max - h_min == (height - 1) and w_max - w_min == (width - 1):
            return img, False
        else:
            current_prob = random.random()
            if aug_prob < current_prob:
                return img, False

            candidate_list = [(h_min - 0) * width, (w_min - 0) * height, (width - w_max - 1) * height, (height - h_max - 1) * width]
            max_idx = candidate_list.index(max(candidate_list))

            bg_only_img = torch.zeros_like(img)
            if candidate_list[max_idx] <= 0:
                raise ValueError
            elif max_idx == 0:
                nb_y = int(height / h_min)
                bg_only_img[:, 0:h_min*nb_y, :] = torch.tile(img[:, 0:h_min, :], (1, nb_y, 1))
                bg_only_img[:, h_min*nb_y:height, :] = img[:, 0:(height - h_min*nb_y), :]
            elif max_idx == 1:
                nb_x = int(width / w_min)
                bg_only_img[:, :, 0:w_min*nb_x] = torch.tile(img[:, :, 0:w_min], (1, 1, nb_x))
                bg_only_img[:, :, w_min*nb_x:width] = img[:, :, 0:(width - w_min*nb_x)]
            elif max_idx == 2:
                nb_x = int(width / (width - w_max))
                bg_only_img[:, :, 0:(width - w_max)*nb_x] = torch.tile(img[:, :, w_max:width], (1, 1, nb_x))
                bg_only_img[:, :, (width - w_max)*nb_x:width] = img[:, :, w_max:w_max + width - (width - w_max)*nb_x]
            else:
                nb_y = int(height / (height - h_max))
                bg_only_img[:, 0:(height- h_max)*nb_y, :] = torch.tile(img[:, h_max:height, :], (1, nb_y, 1))
                bg_only_img[:, (height - h_max)*nb_y:height, :] = img[:, h_max:h_max + height - (height - h_max)*nb_y, :]

            rectangle_mask = torch.zeros_like(mask)
            rectangle_mask[h_min:h_max, w_min:w_max] = 1
            bg_only_img = torch.where(
                rectangle_mask > 0,
                bg_only_img, img)
            return bg_only_img, True

    def bg_mixup(self, img1, img2, mask1):
        bg_mixup_img = mask1 * img1 + (1. - mask1) * img2
        return bg_mixup_img

    def generate_bg_mixed_img(self, img, masks1, cam_thres, aug_prob):
        hard_masks1 = torch.zeros_like(masks1)
        hard_masks1[masks1 >= cam_thres] = 1.

        b, c, img_h, img_w = img.size()
        rand_index = torch.randperm(b).cuda()

        bg_only_imgs = []
        does_bg_exist = [] # check whether the minimum rectangle inlcuding maskment is whole img or not
        for i in range(b):
            h_min, h_max, w_min, w_max = self.get_minimum_rectangle_including_mask(hard_masks1[i])
            bg_only_img, bg_existance = self.generate_bg_only_img(img[i], hard_masks1[i], h_min, h_max, w_min, w_max, aug_prob)
            bg_only_imgs.append(bg_only_img.cuda())
            does_bg_exist.append(bg_existance)

        bg_mixed_imgs = []
        for i in range(b):
            if not does_bg_exist[i]:
                # if the background does not exist, just use original image
                bg_mixed_imgs.append(img[i])
            else:
                if not does_bg_exist[rand_index[i]]:
                    if len(self.bg_queue) == 0:
                        bg_mixed_imgs.append(img[i])

                    else:
                        target_img = self.bg_queue.get()
                        bg_mixed_img = self.bg_mixup(img[i], target_img, masks1[i])
                        bg_mixed_imgs.append(bg_mixed_img)
                else:
                    target_img = bg_only_imgs[rand_index[i]]
                    bg_mixed_img = self.bg_mixup(img[i], target_img, masks1[i])
                    bg_mixed_imgs.append(bg_mixed_img)

            if does_bg_exist[i]:
                self.bg_queue.push(bg_only_imgs[i])

        return torch.stack(bg_mixed_imgs), torch.stack(bg_only_imgs)
