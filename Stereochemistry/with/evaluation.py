import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop
from tqdm.auto import tqdm
import time
from models import EncoderDecodertrain18
from dataset import GetDataset, CapsCollate
from vocab import Vocabulary
import matplotlib.pyplot as plt
import Levenshtein
import json
from utils import get_best_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tqdm.pandas()


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


if __name__ == '__main__':
    dataset_name = 'data'

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

    train_dataset = GetDataset(transform, dataset_name + '/label/train_label.csv', 'train', vocab)
    test_dataset = GetDataset(transform, dataset_name + '/label/test_label.csv', 'test', vocab)

    pad_idx = vocab.stoi["<pad>"]

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=10,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True),
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

    f = open('result/' + dataset_name + '.txt', 'w')

    count = 0
    print(len(test_dataset))
    model.eval()
    with torch.no_grad():
        for i, (image, captions) in enumerate(test_dataloader):
            img, target = image.to(device), captions.to(device)
            features = model.encoder(img)
            caps = model.decoder.generate_caption_batch(features, stoi=vocab.stoi, itos=vocab.itos)
            gen_captions = vocab.tensor_to_captions(caps)
            targets = vocab.tensor_to_captions(target)

            same_list = [x for _, x in enumerate(gen_captions) if x == targets[_]]
            count += len(same_list)
            print('Process: [{}/{}]\tAcc {}'.format(i, len(test_dataloader), len(same_list) / len(target)))

            for _, item in enumerate(gen_captions):
                f.write(item + '\n')
    f.close()
