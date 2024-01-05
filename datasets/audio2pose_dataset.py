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

class StyleDataset_pose_mead_hdtf(Dataset):
    def __init__(self, root_dir, root_dir_HDTF, is_train=True):
        self.root_dir = root_dir
        self.root_dir_HDTF = root_dir_HDTF
        self.is_train = is_train
        if self.is_train:
            file = 'lists/train.json'
            with open(file,"r") as f:
                self.video_list = json.load(f)
            file_HDTF = 'HDTF/lists/train.json'
            with open(file_HDTF,"r") as f:
                self.video_list += json.load(f)
        else:
            file = 'lists/test.json'
            with open(file,"r") as f:
                self.video_list = json.load(f)

            file_HDTF = 'HDTF/lists/test.json'
            with open(file_HDTF,"r") as f:
                self.video_list += json.load(f)
            file_HDTF = 'HDTF/lists/val.json'
            with open(file_HDTF,"r") as f:
                self.video_list += json.load(f)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        name = self.video_list[idx]
        if len(name.split('#'))==1:
            a,b,c = name.split('_')
            npy_path = os.path.join(self.root_dir, a, 'video_25', b, c+'_coeff_pt.npy')
            coeffs_pred_numpy = np.load(npy_path, allow_pickle=True)
            coeffs_pred_numpy = dict(enumerate(coeffs_pred_numpy.flatten(), 0))[0]
            coeff = coeffs_pred_numpy['coeff']
            coeff_mouth = np.concatenate([coeff[:,224:227],coeff[:,254:257]],1)

            
            while(len(coeff_mouth)<50):
                coeff_mouth = np.concatenate((coeff_mouth,coeff_mouth[::-1,:]),axis = 0)

            len_ = len(coeff_mouth)
            r = random.choice([x for x in range(3,len_-32)])
            coeff_mouth = coeff_mouth[r:r+32,:]

            parameters = np.array(coeff_mouth)

            # label = int(self.labels.index(a+'_'+b))
            return parameters# , label
        else:
            if self.is_train:
                file_dir = os.path.join(self.root_dir_HDTF, 'train')
            else:
                file_dir = os.path.join(self.root_dir_HDTF, 'test')
            npy_path = os.path.join(file_dir, name+'.npy')
            coeffs_pred_numpy = np.load(npy_path, allow_pickle=True)
            coeffs_pred_numpy = dict(enumerate(coeffs_pred_numpy.flatten(), 0))[0]
            coeff = coeffs_pred_numpy['coeff']
            coeff_mouth = np.concatenate([coeff[:,224:227],coeff[:,254:257]],1)

            
            while(len(coeff_mouth)<50):
                coeff_mouth = np.concatenate((coeff_mouth,coeff_mouth[::-1,:]),axis = 0)

            len_ = len(coeff_mouth)
            r = random.choice([x for x in range(3,len_-32)])
            coeff_mouth = coeff_mouth[r:r+32,:]

            parameters = np.array(coeff_mouth)

            # label = int(self.labels.index(a+'_'+b))
            return parameters# , label   


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

import lmdb

class StyleDataset_audio_driven_pose(Dataset):
    def __init__(self, mead, hdtf,lmdb_path='SAAS_lmdb', is_train=True):
        self.is_train = is_train
        self.mead = mead
        self.hdtf = hdtf
        
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
        label_file = 'lists/label.json'
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
            while(1):
                key = format_for_lmdb(name, 'length')
                len_ = int(txn.get(key).decode('utf-8'))
                coeff_mouth_key = format_for_lmdb(name, 'coeff_3dmm') # M027#neutral#014-0000153
                coeff_mouths = np.frombuffer(txn.get(coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                len_ = min(len_, len(coeff_mouths))
                if len_>=70:
                    break
                idx = random.choice([x for x in range(len(self.video_list))])
                name = self.video_list[idx]
            splits = name.split('#')
            if len(splits) != 2:
                coeff_mouth_key = format_for_lmdb(name, 'coeff_3dmm') # M027#neutral#014-0000153
                coeff_mouths = np.frombuffer(txn.get(coeff_mouth_key), dtype=np.float32).reshape((-1,257))
                a,b,c,d = splits

                audio_path = os.path.join(self.mead,a,'mfcc',b,c,d+'.npy')
                audio_feature = np.load(audio_path)[:, :, 1:]

                if len_ == 64:
                    r = 0
                else:
                    r = random.choice([x for x in range(len_-64)])
                coeff_mouth = coeff_mouths[r:r+64,:]
                audio_feature = audio_feature[r:(r+64),:]
                parameters = np.array(coeff_mouth)

                label = int(self.labels.index(splits[0]+'#'+splits[1]+'#'+splits[2]))
    
            else:
                a,b = splits

                coeff_mouth_key = format_for_lmdb(name, 'coeff_3dmm') # M027#neutral#014-0000153
                coeff_mouths = np.frombuffer(txn.get(coeff_mouth_key), dtype=np.float32).reshape((-1,257))

                audio_path = os.path.join(self.hdtf,'mfcc',name+'.npy')
                audio_feature = np.load(audio_path)[:, :, 1:]

                if len_ == 64:
                    r = 0
                else:
                    r = random.choice([x for x in range(len_-64)])
                coeff_mouth = coeff_mouths[r:r+64,:]
                
                audio_feature = audio_feature[r:(r+64),:]
                parameters = np.array(coeff_mouth)
                label = int(self.labels.index(splits[0]))

            return parameters, label, name,audio_feature
