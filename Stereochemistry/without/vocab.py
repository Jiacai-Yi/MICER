# -*- coding: utf-8 -*-
"""
# @Time    : 2021/3/9 下午10:26
# @Author  : Jiacai Yi
# @FileName: vocab.py
# @E-mail  ：1076365758@qq.com
"""
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold):
        # setting the pre-reserved tokens int to string tokens
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>"}

        # string to int tokens
        # its reverse dict self.itos
        self.stoi = {v: k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, smiles_list):
        frequencies = Counter()
        idx = 3

        for smiles in smiles_list:
            for char in (smiles):
                frequencies[char] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[char] == self.freq_threshold:
                    self.stoi[char] = idx
                    self.itos[idx] = char
                    idx += 1
        # print(self.stoi)
        # print(self.itos)

    def string_to_ints(self, string):
        l = [self.stoi['<sos>']]
        for s in string:
            l.append(self.stoi[s])
        l.append(self.stoi['<eos>'])
        return l

    def ints_to_string(self, l):
        return ''.join(list(map(lambda i: self.itos[i], l)))

    def tensor_to_captions(self, ten):
        l = ten.tolist()
        ret = []
        for ls in l:
            temp = ''
            # for i in ls[1:]:
            for i in ls:
                if i == self.stoi['<sos>']:
                    continue
                if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                    break
                temp = temp + self.itos[i]
            ret.append(temp)
        return ret

# words = set()
# for st in train['smiles']:
#     words.update(set(st))
#
# vocab = sorted(list(words))
# vocab.append('<sos>')
# vocab.append('<eos>')
# vocab.append('<pad>')
#
# stoi = dict()
# for _, item in enumerate(vocab):
#     stoi[item] = _
#
# itos = {item[1]: item[0] for item in stoi.items()}
