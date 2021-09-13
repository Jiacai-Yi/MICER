# -*- coding: utf-8 -*-
"""
# @Time    : 2021/3/9 下午8:51
# @Author  : Jiacai Yi
# @FileName: models.py
# @E-mail  ：1076365758@qq.com
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm

tqdm.pandas()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,64,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size,64,attemtion_dim)

        attention_scores = self.A(combined_states)  # (batch_size,64,1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size,64)

        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,64)

        attention_weights = features * alpha.unsqueeze(2)  # (batch_size,64,features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size,64)

        return alpha, attention_weights


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()

        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):

        # vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # get the seq length to iterate
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def generate_caption(self, features, max_len=202, itos=None, stoi=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # starting input
        word = torch.tensor(stoi['<sos>']).view(1, -1).to(device)
        # word = torch.full((batch_size, 1), stoi['<sos>']).to(device)
        embeds = self.embedding(word)

        captions = []
        # captions = torch.zeros((batch_size, 202), dtype=torch.long).to(device)
        # captions[:, 0] = word.squeeze()

        for i in range(max_len):
            alpha, context = self.attention(features, h)

            # store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            # print('embeds',embeds.shape)
            # print('embeds[:,0]',embeds[:,0].shape)
            # print('context',context.shape)
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            # print('output',output.shape)
            output = output.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word_idx.item())
            # print('predicted_word_idx',predicted_word_idx.shape)
            # captions[:, i] = predicted_word_idx

            # end if <EOS detected>
            if itos[predicted_word_idx.item()] == "<eos>":
                break

            # send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
            # embeds = self.embedding(predicted_word_idx).unsqueeze(1)

        # covert the vocab idx to words and return sentence
        return [itos[idx] for idx in captions], alphas
        # return captions

    def generate_caption_batch(self, features, max_len=202, itos=None, stoi=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # starting input
        # word = torch.tensor(stoi['<sos>']).view(1,-1).to(device)
        word = torch.full((batch_size, 1), stoi['<sos>']).to(device)
        embeds = self.embedding(word)

        # captions = []
        captions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)
        captions[:, 0] = word.squeeze()

        for i in range(max_len):
            alpha, context = self.attention(features, h)

            # store the apla score
            # alphas.append(alpha.cpu().detach().numpy())
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            # captions.append(predicted_word_idx.item())
            captions[:, i] = predicted_word_idx

            # end if <EOS detected>
            # if itos[predicted_word_idx.item()] == "<eos>":
            #    break

            # send generated word as the next caption
            # embeds = self.embedding(predicted_word_idx.unsqueeze(0))
            embeds = self.embedding(predicted_word_idx).unsqueeze(1)

            # covert the vocab idx to words and return sentence
            # return [itos[idx] for idx in captions]
        return captions

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


class EncoderCNNtrain18(nn.Module):
    def __init__(self):
        super(EncoderCNNtrain18, self).__init__()
        resnet = torchvision.models.resnet18()
        # for param in resnet.parameters():
        #     param.requires_grad = False

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,512,8,8)
        features = features.permute(0, 2, 3, 1)  # (batch_size,8,8,512)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,64,512)
        return features


class EncoderDecodertrain18(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNNtrain18()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
