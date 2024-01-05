import os
from unittest import main
from skimage import io, img_as_float32, transform
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import glob
import pickle
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch, json


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

import lmdb

class StyleDataset_video_driven(Dataset):
    def __init__(self, lmdb_path='SAAS_lmdb', is_train=True):
        self.is_train = is_train
        
        if self.is_train:
            self.type = 'train'
            file = 'lists/train.json'
            with open(file,"r") as f:
                self.video_list = json.load(f)
        else:
            self.type = 'test'
            file = 'lists/test.json'
            with open(file,"r") as f:
                self.video_list = json.load(f)
        random.shuffle(self.video_list)
        label_file = '/label.json'
        with open(label_file,"r") as f:
            self.labels = json.load(f)
        self.ids = {}
        for name in self.video_list:
            splits = name.split('#')
            if len(splits) == 2:
                key = splits[0]
            else:
                key = splits[0]+'#'+splits[1]+'#'+splits[2]

            if key in self.ids.keys():
                self.ids[key].append(name)
            else:
                self.ids[key] = [name]
        self.keys = self.ids.keys()

        self.env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_path)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            name = self.video_list[idx]
            splits = name.split('#')
            if len(splits) != 2:
                coeff_mouth_key = format_for_lmdb(name, 'coeff_3dmm') # M027#neutral#014-0000153
                coeff_mouth = np.frombuffer(txn.get(coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                key = format_for_lmdb(name, 'length')
                len_ = int(txn.get(key).decode('utf-8'))
                a,b,c,d = splits

                if len_ == 32:
                    r = 0
                else:
                    r = random.choice([x for x in range(len_-32)])
                coeff_mouth = coeff_mouth[r:r+32,:]

                parameters = np.array(coeff_mouth)

                label = int(self.labels.index(splits[0]+'#'+splits[1]+'#'+splits[2]))

                style_index = random.choice([x for x in range(len(self.video_list))])

                style_name = self.video_list[style_index]
                style_a = style_name.split('#')[0]
                while (style_a==a):
                    style_index = random.choice([x for x in range(len(self.video_list))])
                    style_name = self.video_list[style_index]
                    style_a = style_name.split('#')[0]

                style_splits = style_name.split('#')
                style_splits = style_name.split('#')
                if len(style_splits) == 2:
                    style_a, style_b = style_splits
                    style_label = int(self.labels.index(style_a))
                else:
                    style_a,style_b,style_c,style_d = style_splits
                    style_label = int(self.labels.index(style_a+'#'+style_b+'#'+style_c))

                style_coeff_mouth_key = format_for_lmdb(style_name, 'coeff_3dmm') # M027#neutral#014-0000153
                style_coeff_mouth = np.frombuffer(txn.get(style_coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                style_key = format_for_lmdb(style_name, 'length')
                style_len_ = int(txn.get(style_key).decode('utf-8'))

                if style_len_ == 32:
                    style_r = 0
                else:
                    style_r = random.choice([x for x in range(style_len_-32)])

                style_coeff_mouth = style_coeff_mouth[style_r:style_r+32,:]

                style_parameters = np.array(style_coeff_mouth)


                rr = random.randint(0,len(self.ids[a+'#'+b+'#'+c])-1)
                positive_name = self.ids[a+'#'+b+'#'+c][rr]
                positive_a, positive_b, positive_c, positive_d = positive_name.split('#')

                positive_coeff_mouth_key = format_for_lmdb(positive_name, 'coeff_3dmm') # M027#neutral#014-0000153
                positive_coeff_mouth = np.frombuffer(txn.get(positive_coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                positive_key = format_for_lmdb(positive_name, 'length')
                positive_len_ = int(txn.get(positive_key).decode('utf-8'))

                if positive_len_ == 32:
                    positive_r = 0
                else:
                    positive_r = random.choice([x for x in range(positive_len_-32)])


                positive_coeff_mouth = positive_coeff_mouth[positive_r:positive_r+32,:]
                positive_parameters = np.array(positive_coeff_mouth)

                emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']

                negetive_r = random.randint(0,6)
                while b == emotions[negetive_r] or a+'#'+emotions[negetive_r]+'#level_3' not in self.keys:
                    negetive_r = random.randint(0,6)


                rr = random.randint(0,len(self.ids[a+'#'+emotions[negetive_r]+'#level_3'])-1)
                negetive_name = self.ids[a+'#'+emotions[negetive_r]+'#level_3'][rr]
                negetive_a, negetive_b, negetive_c, negetive_d = negetive_name.split('#')

                negetive_coeff_mouth_key = format_for_lmdb(negetive_name, 'coeff_3dmm') # M027#neutral#014-0000153
                negetive_coeff_mouth = np.frombuffer(txn.get(negetive_coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                negetive_key = format_for_lmdb(negetive_name, 'length')
                negetive_len_ = int(txn.get(negetive_key).decode('utf-8'))


                if negetive_len_ == 32:
                    negetive_r = 0
                else:
                    negetive_r = random.choice([x for x in range(negetive_len_-32)])


                negetive_coeff_mouth = negetive_coeff_mouth[negetive_r:negetive_r+32,:]
                negetive_parameters = np.array(negetive_coeff_mouth)
            else:
                a,b = splits

                coeff_mouth_key = format_for_lmdb(name, 'coeff_3dmm') # M027#neutral#014-0000153
                coeff_mouth = np.frombuffer(txn.get(coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                key = format_for_lmdb(name, 'length')
                len_ = int(txn.get(key).decode('utf-8'))
                
                if len_ == 32:
                    r = 0
                else:
                    r = random.choice([x for x in range(len_-32)])
                coeff_mouth = coeff_mouth[r:r+32,:]

                parameters = np.array(coeff_mouth)

                label = int(self.labels.index(splits[0]))

                style_index = random.choice([x for x in range(len(self.video_list))])

                style_name = self.video_list[style_index]
                style_a = style_name.split('#')[0]
                while (style_a==a):
                    style_index = random.choice([x for x in range(len(self.video_list))])
                    style_name = self.video_list[style_index]
                    style_a = style_name.split('#')[0]

                style_splits = style_name.split('#')

                if len(style_splits) == 2:
                    style_a, style_b = style_splits
                    style_label = int(self.labels.index(style_a))
                else:
                    style_a,style_b,style_c,style_d = style_splits
                    style_label = int(self.labels.index(style_a+'#'+style_b+'#'+style_c))


                style_coeff_mouth_key = format_for_lmdb(style_name, 'coeff_3dmm') # M027#neutral#014-0000153
                style_coeff_mouth = np.frombuffer(txn.get(style_coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                style_key = format_for_lmdb(style_name, 'length')
                style_len_ = int(txn.get(style_key).decode('utf-8'))

                if style_len_ == 32:
                    style_r = 0
                else:
                    style_r = random.choice([x for x in range(style_len_-32)])



                style_coeff_mouth = style_coeff_mouth[style_r:style_r+32,:]

                style_parameters = np.array(style_coeff_mouth)


                rr = random.randint(0,len(self.ids[a])-1)
                positive_name = self.ids[a][rr]
                positive_coeff_mouth_key = format_for_lmdb(positive_name, 'coeff_3dmm') # M027#neutral#014-0000153
                positive_coeff_mouth = np.frombuffer(txn.get(positive_coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                positive_key = format_for_lmdb(positive_name, 'length')
                positive_len_ = int(txn.get(positive_key).decode('utf-8'))

                if positive_len_ == 32:
                    positive_r = 0
                else:
                    positive_r = random.choice([x for x in range(positive_len_-32)])


                positive_coeff_mouth = positive_coeff_mouth[positive_r:positive_r+32,:]
                positive_parameters = np.array(positive_coeff_mouth)

                negetive_index = random.choice([x for x in range(len(self.video_list))])

                negetive_name = self.video_list[negetive_index]
                
                negetive_a = negetive_name.split('#')[0]
                while (negetive_a==a):
                    negetive_index = random.choice([x for x in range(len(self.video_list))])
                    negetive_name = self.video_list[negetive_index]
                    negetive_a = negetive_name.split('#')[0]


                negetive_coeff_mouth_key = format_for_lmdb(negetive_name, 'coeff_3dmm') # M027#neutral#014-0000153
                negetive_coeff_mouth = np.frombuffer(txn.get(negetive_coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                negetive_key = format_for_lmdb(negetive_name, 'length')
                negetive_len_ = int(txn.get(negetive_key).decode('utf-8'))


                if negetive_len_ == 32:
                    negetive_r = 0
                else:
                    negetive_r = random.choice([x for x in range(negetive_len_-32)])

                negetive_coeff_mouth = negetive_coeff_mouth[negetive_r:negetive_r+32,:]

                negetive_parameters = np.array(negetive_coeff_mouth)
            

            # print(name,len(parameters),r,len_,style_name,len(style_parameters),style_r,style_len_,positive_name,len(positive_parameters),positive_r,positive_len_,negetive_name,len(negetive_parameters),negetive_r,negetive_len_)
            while len(parameters)<32:
                parameters = np.concatenate((parameters, parameters[-1:]),0)
            while len(style_parameters)<32:
                style_parameters = np.concatenate((style_parameters, style_parameters[-1:]),0)
            while len(positive_parameters)<32:
                positive_parameters = np.concatenate((positive_parameters, positive_parameters[-1:]),0)
            while len(negetive_parameters)<32:
                negetive_parameters = np.concatenate((negetive_parameters, negetive_parameters[-1:]),0)
            
            return  style_parameters, style_label, parameters, label, positive_parameters, negetive_parameters


