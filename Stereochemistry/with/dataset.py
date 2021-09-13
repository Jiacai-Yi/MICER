# -*- coding: utf-8 -*-
"""
# @Time    : 2021/3/9 下午8:54
# @Author  : Jiacai Yi
# @FileName: dataset.py
# @E-mail  ：1076365758@qq.com
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
from PIL import Image


def get_train_file_path(image_id):
    dir_index = image_id // 64000
    return "data/train/{}/{}.png".format(
        dir_index, image_id
    )


def get_test_file_path(image_id):
    return "data/test/{}.png".format(
        image_id
    )


def get_val_file_path(image_id):
    return "data/val/{}.png".format(
        image_id
    )


def pil_loader(path: str) -> Image.Image:  # copied from torchvision
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if not os.path.exists(path) or not os.path.getsize(path):
        return None
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class GetDataset(Dataset):
    def __init__(self, transform, caption_file, type='train', vocab=None):
        self.loader = default_loader
        self.transform = transform
        self.caption_file = pd.read_csv(caption_file)
        if type == 'train':
            self.caption_file['file_path'] = self.caption_file['image_id'].progress_apply(get_train_file_path)
        elif type == 'test':
            self.caption_file['file_path'] = self.caption_file['image_id'].progress_apply(get_test_file_path)
        elif type == 'val':
            self.caption_file['file_path'] = self.caption_file['image_id'].progress_apply(get_val_file_path)
        self.vocab = vocab

    def __len__(self):
        return len(self.caption_file)

    def __getitem__(self, idx):
        sample = self.loader(self.caption_file['file_path'][idx])
        if sample:
            sample = self.transform(sample)

        caption_vec = []
        # caption_vec += [self.vocab.stoi["<sos>"]]
        caption_vec += self.vocab.string_to_ints(self.caption_file['smiles'][idx])
        # caption_vec += [self.vocab.stoi["<eos>"]]

        # return sample, caption, idx

        return sample, torch.tensor(caption_vec)


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """

    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = []
        targets = []

        for item in batch:
            imgs.append(item[0].unsqueeze(0))
            targets.append(item[1])
        imgs = torch.cat(imgs, dim=0)
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets
