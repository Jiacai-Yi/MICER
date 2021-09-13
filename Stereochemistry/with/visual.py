# -*- coding: utf-8 -*-
"""
# @Time    : 2021/3/13 上午9:32
# @Author  : Jiacai Yi
# @FileName: visual.py
# @E-mail  ：1076365758@qq.com
"""
import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop
from models import EncoderDecodertrain18
from dataset import GetDataset, CapsCollate
from vocab import Vocabulary
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tqdm.pandas()

dataset = 'data'


def show_image(img, title=None):
    """Imshow for Tensor."""

    # unnormalize
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# generate caption
def get_caps_from(model, features_tensors, vocab):
    # generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, stoi=vocab.stoi,
                                                      itos=vocab.itos)
        caption = ' '.join(caps)
        show_image(features_tensors[0], title=caption)

    return caps, alphas


# Show attention
def plot_attention(img, result, attention_plot, target):
    # untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    color = 'YlGnBu'
    alpha_all = 0.4
    fontsize = 40

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img
    # fig = plt.figure(figsize=(32, 32))
    fig = plt.figure(figsize=(32, 32))

    result = result[:-1]

    len_result = len(result)
    all_att = np.zeros(shape=(8, 8))

    ax = fig.add_subplot(3, 5, 1)
    ax.set_title('Truth\n' + ''.join(target), fontsize=fontsize)
    img = ax.imshow(temp_image)
    ax.imshow(np.ones(shape=(8, 8)), cmap=color, alpha=0, extent=img.get_extent())
    plt.axis('off')
    # cbar_kw = {}
    # cbar = ax.figure.colorbar(img, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel('attention weight', rotation=-90, va="bottom")

    for l in range(len_result):
        temp_att = attention_plot[l].reshape(8, 8)
        ax = fig.add_subplot(3, 5, l + 2)
        ax.set_title(result[l], fontsize=fontsize)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap=color, alpha=alpha_all, extent=img.get_extent())
        plt.axis('off')
        all_att += temp_att

    ax = fig.add_subplot(3, 5, len_result + 2)
    ax.set_title('Predicted\n' + ''.join(result), fontsize=fontsize)
    img = ax.imshow(temp_image)
    ax.imshow(all_att, cmap=color, alpha=alpha_all, extent=img.get_extent())
    plt.axis('off')

    plt.tight_layout()
    plt.show()


import json


def get_best_model(dir):
    best_acc = -float('inf')
    best_model_path = ''
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_type = os.path.splitext(file)[1]
            if file_type == '.pth':
                MODEL_PATH = os.path.join(root, file)
                model_state = torch.load(MODEL_PATH)
                print(model_state['val_acc'])
                if model_state['val_acc'] > best_acc:
                    best_acc = model_state['val_acc']
                    best_model_path = os.path.join(root, file)
    return best_model_path


if __name__ == '__main__':
    transform = Compose([
        # RandomHorizontalFlip(),
        # Resize((256, 256), PIL.Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    vocab = Vocabulary(5)
    with open('tmp_files/word_2_index.json', 'r') as f:
        load_dict = json.load(f)
    vocab.stoi = load_dict
    vocab.itos = {v: k for k, v in vocab.stoi.items()}

    test_dataset = GetDataset(transform, dataset + '/label/test_label.csv', 'test', vocab)

    pad_idx = vocab.stoi["<pad>"]

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        # shuffle=True,
        num_workers=10,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )

    MODEL_PATH = get_best_model('model_files')
    model_state = torch.load(MODEL_PATH)
    model = EncoderDecodertrain18(
        embed_size=model_state['embed_size'],
        vocab_size=model_state['vocab_size'],
        attention_dim=model_state['attention_dim'],
        encoder_dim=model_state['encoder_dim'],
        decoder_dim=model_state['decoder_dim']
    ).to(device)
    model.load_state_dict(model_state['state_dict'])
    model = model.to(device)

    # show any 1
    dataiter = iter(test_dataloader)
    # images, _ = None, None
    flag = True
    while True:
        images, _ = next(dataiter)
        target = vocab.tensor_to_captions(_)
        for index, item in enumerate(target):
            # if len(item) == 13:
            img = images[index].detach().clone()
            img1 = images[index].detach().clone()
            caps, alphas = get_caps_from(model, img.unsqueeze(0), vocab)
            plot_attention(img1, caps, alphas, target[index])
            flag = False
            break
