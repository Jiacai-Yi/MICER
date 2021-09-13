import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL
import time
import matplotlib.pyplot as plt
import json
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn.functional as F
import datetime
import seaborn as sns
from tqdm.auto import tqdm
from models import EncoderDecodertrain18
from dataset import GetDataset, CapsCollate
from vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tqdm.pandas()

vocab = Vocabulary(5)
with open('tmp_files/word_2_index.json', 'r') as f:
    load_dict = json.load(f)
vocab.stoi = load_dict
vocab.itos = {v: k for k, v in vocab.stoi.items()}


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


transform = Compose([
    # RandomHorizontalFlip(),
    # Resize((256, 256), PIL.Image.BICUBIC),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = GetDataset(transform, 'data2/label/train_label.csv', 'train', vocab)
val_dataset = GetDataset(transform, 'data2/label/val_label.csv', 'val', vocab)

pad_idx = vocab.stoi["<pad>"]

batch_size = 128

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

embed_size = 300
attention_dim = 256
encoder_dim = 512
decoder_dim = 512
learning_rate = 3e-5
vocab_size = len(vocab)

model = EncoderDecodertrain18(
    embed_size=embed_size,
    vocab_size=vocab_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim
).to(device)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


data_time = AverageMeter()  # data loading time
accs = AverageMeter()  # top5 accuracy

criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def save_model(model, num_epochs, acc):
    model_state = {
        'num_epochs': num_epochs,
        'embed_size': embed_size,
        'vocab_size': len(vocab),
        'attention_dim': attention_dim,
        'encoder_dim': encoder_dim,
        'decoder_dim': decoder_dim,
        'state_dict': model.state_dict(),
        'val_acc': acc
    }

    # torch.save(model_state,
    #            'model_files/2021-03-18/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_attention_model_state_' + str(
    #                num_epochs) + '.pth')

    torch.save(model_state,
               'model_files/' + 'attention_model_state_' + str(
                   num_epochs) + '.pth')


num_epochs = 10
print_every = 20
print_every_img = 1000


def accuracy(outputs, targets):
    num_correct = 0
    pred = outputs.argmax(dim=-1)
    pred = pred.cpu().numpy()
    targets = targets.cpu().numpy()
    pred_list = np.array([])
    target_list = np.array([])

    for _, item in enumerate(pred):
        np.append(pred_list, ','.join([str(i) for i in item.tolist()]))
    for _, item in enumerate(targets):
        np.append(target_list, ','.join([str(i) for i in item.tolist()]))

    correct_cnt = torch.eq(torch.from_numpy(pred_list), torch.from_numpy(target_list)).view(-1).float().sum().item()
    # num_correct += torch.eq(pred, targets).sum().float().item()

    return correct_cnt / len(pred_list)


for epoch in range(1, num_epochs + 1):
    for idx, (image, captions) in enumerate(iter(train_dataloader)):
        image, captions = image.to(device), captions.to(device)

        # show_image(image[0].cpu(), title=train_dataset.vocab.ints_to_string(captions[0].cpu().numpy().tolist()))

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs, attentions = model(image, captions)

        # Calculate the batch loss.
        targets = captions[:, 1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # acc = accuracy(outputs, targets)

        if (idx + 1) % print_every == 0:
            # print('Epoch: [{}][{}/{}]\tLoss {}\tAcc {}'.format(epoch, idx, len(train_dataloader), loss.item(),
            #                                                    acc.item()))
            print('Epoch: [{}][{}/{}]\tLoss {}'.format(epoch, idx, len(train_dataloader), loss.item()))

        if (idx + 1) % print_every_img == 0:
            # generate the caption
            model.eval()
            with torch.no_grad():
                dataiter = iter(train_dataloader)
                img, _ = next(dataiter)
                features = model.encoder(img[0:1].to(device))
                caps, alphas = model.decoder.generate_caption(features, stoi=vocab.stoi,
                                                              itos=vocab.itos)
                caption = ' '.join(caps)
                show_image(img[0], title=caption)

            model.train()

    torch.cuda.empty_cache()

    # one epoch validation
    count = 0
    model.eval()
    with torch.no_grad():
        for _, (image, captions) in enumerate(val_dataloader):
            # Move to device, if available
            img, target = image.to(device), captions.to(device)
            features = model.encoder(img)
            caps = model.decoder.generate_caption_batch(features, stoi=vocab.stoi, itos=vocab.itos)
            gen_captions = vocab.tensor_to_captions(caps)
            targets = vocab.tensor_to_captions(target)
            same_list = [x for _, x in enumerate(gen_captions) if x == targets[_]]
            count += len(same_list)
            if (_ + 1) % print_every == 0:
                print('Epoch: [{}][{}/{}\tAcc {}'.format(epoch, _, len(val_dataloader),
                                                         len(same_list) / len(target)))

    model.train()
    val_acc = count / len(val_dataset)
    print(
        '\n * SEQUENCE ACCURACY - {acc:.3f} * \n'.format(
            acc=val_acc))

    # save the latest model
    save_model(model, epoch, val_acc)
    torch.cuda.empty_cache()
