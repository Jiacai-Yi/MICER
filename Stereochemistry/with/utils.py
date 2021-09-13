# -*- coding: utf-8 -*-
"""
# @Time    : 2021/3/9 下午2:30
# @Author  : Jiacai Yi
# @FileName: utils.py
# @E-mail  ：1076365758@qq.com
"""

import pandas as pd
from indigo import IndigoObject, Indigo
from indigo.renderer import IndigoRenderer
import os
from tqdm import tqdm
import torch


def get_img(_, smiles, path):
    # generate images
    indigo = Indigo()
    indigo_render = IndigoRenderer(indigo)
    indigo.setOption('render-image-width', 256)
    indigo.setOption('render-image-height', 256)
    indigo.setOption('render-background-color', "1,1,1")
    indigo.setOption('render-stereo-style', 'none')

    # new
    indigo.setOption('render-implicit-hydrogens-visible', False)

    indigo_render.renderToFile(indigo.loadMolecule(smiles), path + str(_) + '.png')


def get_dataset():
    all_smiles = []
    with open('tmp_smiles2img_files/last_train_smiles.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            all_smiles.append(line)
    for _, smiles in tqdm(enumerate(all_smiles)):
        get_img(_, smiles, 'data/train/' + str(_ // 72000) + '/')

    all_test_smiles = []
    with open('tmp_smiles2img_files/last_test_smiles.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            all_test_smiles.append(line)
    for _, smiles in tqdm(enumerate(all_test_smiles)):
        get_img(_, smiles, 'data/test/')


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


def statistics():
    path = 'data2/label/test_label.csv'
    f = open('statistic_file.txt', 'w')
    datas = pd.read_csv(path)
    all_smiles = datas['smiles']
    all_count = len(all_smiles)
    invalid_count = 0
    indigo = Indigo()
    all_atom_count = 0
    sequence_length = 0
    max_sq = float("-inf")
    min_sq = float("inf")
    max_atoms = float("-inf")
    min_atoms = float("inf")
    for smiles in all_smiles:
        sequence_length += len(smiles)
        try:
            mol = indigo.loadMolecule(smiles)
        except:
            invalid_count += 1
            continue
        else:
            cnt = (mol.countAtoms() + mol.countStereocenters())
            all_atom_count += cnt
            if cnt > max_atoms:
                max_atoms = cnt
            if cnt < min_atoms:
                min_atoms = cnt
            if len(smiles) > max_sq:
                max_sq = len(smiles)
            if len(smiles) < min_sq:
                min_sq = len(smiles)
    f.write("avg atoms: " + str(all_atom_count / (all_count - invalid_count)) + '\n')
    f.write("max atoms: " + str(max_atoms) + '\n')
    f.write("min atoms: " + str(min_atoms) + '\n')
    f.write("avg sequence length: " + str(sequence_length / all_count) + '\n')
    f.write("max sq: " + str(max_sq) + '\n')
    f.write("min sq: " + str(min_sq) + '\n')
    f.write("\n\n")
    f.close()


from vocab import Vocabulary
import json

if __name__ == '__main__':
    # build dictionary
    caption_file = pd.read_csv('data2/label/train_label.csv')
    vocab = Vocabulary(5)
    vocab.build_vocab(caption_file['smiles'].tolist())
    with open('tmp_files/data2_word_2_index.json', 'w') as f:
        json.dump(vocab.stoi, f)
