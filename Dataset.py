# The code is developed based on SPADE 
# https://github.com/NVlabs/SPADE/

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import re
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp']


def atoi(text):
    return int(text) if text.isdigit() else text


def get_label(text):
    return re.split('(\d+)', re.split("/", text)[-1])[1]


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


##judge whether file is ends with filename
def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    images = []
    # cache :隐藏文件
    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:  # 递归的
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images


def get_params(preprocess_mode, load_size, crop_size, size):
    w, h = size
    new_h = h
    new_w = w
    if preprocess_mode == 'resize_and_crop':
        new_h = new_w = load_size
    elif preprocess_mode == 'scale_width_and_crop':
        new_w = load_size
        new_h = load_size * h // w
    elif preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(params, preprocess_mode='resize_and_crop', load_size=286, crop_size=256, aspect_ratio=1.0, flip=True,
                  method=Image.BILINEAR, normalize=True, toTensor=True, colorjitter=False):
    transform_list = []
    if 'resize' in preprocess_mode:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, method)))
    elif 'scale_shortside' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, load_size, method)))

    if 'crop' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if preprocess_mode == 'fixed':
        w = crop_size
        h = round(crop_size / aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if colorjitter:
        transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.1))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class Cont_Cla_Dataset(data.Dataset):
    def initialize(self, imgroot,files, dataset_sizes,
                   preprocess_mode='resize_and_crop', load_size=286, crop_size=256):
        self.paths = []

        assert len(files) == len(dataset_sizes), \
            "The list number of source image paths, target image paths and dataset sizes don't match."

        for i in range(len(files)):
            paths = self.get_paths(imgroot,files[i])
            dataset_size = dataset_sizes[i]

            natural_sort(paths)
            dataset_size = min(dataset_size, len(paths))
            paths = paths[:dataset_size]

            self.paths += paths

        random.shuffle(self.paths)

        self.dataset_size = len(self.paths)
        self.load_size = load_size
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size

    def get_paths(self, files):
        paths = make_dataset(files, recursive=False, read_cache=True)
        return paths

    def __getitem__(self, index):
        # Label Image
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')

        label = get_label(path)
        params = get_params(self.preprocess_mode, self.load_size, self.crop_size, img.size)
        transform_image = get_transform(params, self.preprocess_mode, self.load_size, self.crop_size)

        Tensor = transform_image(img)

        input_dict = {'img': Tensor,
                      'label': label}
        return input_dict

    def __len__(self):
        return self.dataset_size

def create_con_cla_dataloader(imgroot,files, dataset_sizes, batchSize=16, shuffle=True, nworkers=4,
                   preprocess_mode='resize_and_crop', load_size=286, crop_size=256):
    con_cla_dataset = Cont_Cla_Dataset()
    con_cla_dataset.initialize(imgroot,files ,dataset_sizes, preprocess_mode, load_size, crop_size)
    print("dataset [%s] of size %d was created" %  (type(con_cla_dataset).__name__, len(con_cla_dataset)))
    dataloader = torch.utils.data.DataLoader(
        con_cla_dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=nworkers,
        drop_last=True
    )
    return dataloader

