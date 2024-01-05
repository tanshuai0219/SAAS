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


class StyleDataset_four(Dataset):
    def __init__(self, root_dir,hdtf_dir, is_train=True):
        self.root_dir = root_dir
        self.hdtf_dir = hdtf_dir
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
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        name = self.video_list[idx]
        splits = name.split('#')
        if len(splits) != 2:
            a,b,c,d = splits
            npy_path = os.path.join(self.root_dir, a, '3DMM', b, c,d+'.npy')
            coeffs_pred_numpy = np.load(npy_path, allow_pickle=True)
            coeffs_pred_numpy = dict(enumerate(coeffs_pred_numpy.flatten(), 0))[0]
            coeff = coeffs_pred_numpy['coeff']
            coeff_mouth = coeff[:,80:144]

            
            while(len(coeff_mouth)<50):
                coeff_mouth = np.concatenate((coeff_mouth,coeff_mouth[::-1,:]),axis = 0)
            len_ = len(coeff_mouth)
            r = random.choice([x for x in range(3,len_-32)])
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
            if len(style_splits) == 2:
                style_a, style_b = style_splits
                style_npy_path = os.path.join(self.hdtf_dir, style_name+'.npy')
            else:
                style_a,style_b,style_c,style_d = style_splits
                style_npy_path = os.path.join(self.root_dir, style_a, '3DMM', style_b, style_c,style_d+'.npy')

            style_coeffs_pred_numpy = np.load(style_npy_path, allow_pickle=True)
            style_coeffs_pred_numpy = dict(enumerate(style_coeffs_pred_numpy.flatten(), 0))[0]
            style_coeff = style_coeffs_pred_numpy['coeff']
            style_coeff_mouth = style_coeff[:,80:144]
            
            
            while(len(style_coeff_mouth)<50):
                style_coeff_mouth = np.concatenate((style_coeff_mouth,style_coeff_mouth[::-1,:]),axis = 0)

            style_len_ = len(style_coeff_mouth)
            style_r = random.choice([x for x in range(3,style_len_-32)])
            style_coeff_mouth = style_coeff_mouth[style_r:style_r+32,:]

            style_parameters = np.array(style_coeff_mouth)


            rr = random.randint(0,len(self.ids[a+'#'+b+'#'+c])-1)
            positive_name = self.ids[a+'#'+b+'#'+c][rr]
            positive_a, positive_b, positive_c, positive_d = positive_name.split('#')
            positive_npy_path = os.path.join(self.root_dir, positive_a, '3DMM', positive_b, positive_c,positive_d+'.npy')
            positive_coeffs_pred_numpy = np.load(positive_npy_path, allow_pickle=True)
            positive_coeffs_pred_numpy = dict(enumerate(positive_coeffs_pred_numpy.flatten(), 0))[0]
            positive_coeff = positive_coeffs_pred_numpy['coeff']
            positive_coeff_mouth = positive_coeff[:,80:144]
            while(len(positive_coeff_mouth)<50):
                positive_coeff_mouth = np.concatenate((positive_coeff_mouth,positive_coeff_mouth[::-1,:]),axis = 0)
            positive_len_ = len(positive_coeff_mouth)
            positive_r = random.choice([x for x in range(3,positive_len_-32)])
            positive_coeff_mouth = positive_coeff_mouth[positive_r:positive_r+32,:]
            positive_parameters = np.array(positive_coeff_mouth)

            emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']

            negetive_r = random.randint(0,6)
            while b == emotions[negetive_r] or a+'#'+emotions[negetive_r]+'#level_3' not in self.keys:
                negetive_r = random.randint(0,6)


            rr = random.randint(0,len(self.ids[a+'#'+emotions[negetive_r]+'#level_3'])-1)
            negetive_name = self.ids[a+'#'+emotions[negetive_r]+'#level_3'][rr]
            negetive_a, negetive_b, negetive_c, negetive_d = negetive_name.split('#')
            negetive_npy_path = os.path.join(self.root_dir, negetive_a, '3DMM', negetive_b, negetive_c,negetive_d+'.npy')
            negetive_coeffs_pred_numpy = np.load(negetive_npy_path, allow_pickle=True)
            negetive_coeffs_pred_numpy = dict(enumerate(negetive_coeffs_pred_numpy.flatten(), 0))[0]
            negetive_coeff = negetive_coeffs_pred_numpy['coeff']
            negetive_coeff_mouth = negetive_coeff[:,80:144]
            while(len(negetive_coeff_mouth)<50):
                negetive_coeff_mouth = np.concatenate((negetive_coeff_mouth,negetive_coeff_mouth[::-1,:]),axis = 0)
            negetive_len_ = len(negetive_coeff_mouth)
            negetive_r = random.choice([x for x in range(3,negetive_len_-32)])
            negetive_coeff_mouth = negetive_coeff_mouth[negetive_r:negetive_r+32,:]
            negetive_parameters = np.array(negetive_coeff_mouth)
        else:
            a,b = splits
            npy_path = os.path.join(self.hdtf_dir, name+'.npy') # os.path.join(self.root_dir, a, '3DMM', b, c,d+'.npy')
            coeffs_pred_numpy = np.load(npy_path, allow_pickle=True)
            coeffs_pred_numpy = dict(enumerate(coeffs_pred_numpy.flatten(), 0))[0]
            coeff = coeffs_pred_numpy['coeff']
            coeff_mouth = coeff[:,80:144]

            
            while(len(coeff_mouth)<50):
                coeff_mouth = np.concatenate((coeff_mouth,coeff_mouth[::-1,:]),axis = 0)
            len_ = len(coeff_mouth)
            r = random.choice([x for x in range(3,len_-32)])
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
                style_npy_path = os.path.join(self.hdtf_dir, style_name+'.npy')
            else:
                style_a,style_b,style_c,style_d = style_splits
                style_npy_path = os.path.join(self.root_dir, style_a, '3DMM', style_b, style_c,style_d+'.npy')

            style_coeffs_pred_numpy = np.load(style_npy_path, allow_pickle=True)
            style_coeffs_pred_numpy = dict(enumerate(style_coeffs_pred_numpy.flatten(), 0))[0]
            style_coeff = style_coeffs_pred_numpy['coeff']
            style_coeff_mouth = style_coeff[:,80:144]
            
            
            while(len(style_coeff_mouth)<50):
                style_coeff_mouth = np.concatenate((style_coeff_mouth,style_coeff_mouth[::-1,:]),axis = 0)

            style_len_ = len(style_coeff_mouth)
            style_r = random.choice([x for x in range(3,style_len_-32)])
            style_coeff_mouth = style_coeff_mouth[style_r:style_r+32,:]

            style_parameters = np.array(style_coeff_mouth)


            rr = random.randint(0,len(self.ids[a])-1)
            positive_name = self.ids[a][rr]
            positive_npy_path = os.path.join(self.hdtf_dir, positive_name+'.npy')
            positive_coeffs_pred_numpy = np.load(positive_npy_path, allow_pickle=True)
            positive_coeffs_pred_numpy = dict(enumerate(positive_coeffs_pred_numpy.flatten(), 0))[0]
            positive_coeff = positive_coeffs_pred_numpy['coeff']
            positive_coeff_mouth = positive_coeff[:,80:144]
            while(len(positive_coeff_mouth)<50):
                positive_coeff_mouth = np.concatenate((positive_coeff_mouth,positive_coeff_mouth[::-1,:]),axis = 0)
            positive_len_ = len(positive_coeff_mouth)
            positive_r = random.choice([x for x in range(3,positive_len_-32)])
            positive_coeff_mouth = positive_coeff_mouth[positive_r:positive_r+32,:]
            positive_parameters = np.array(positive_coeff_mouth)

            negetive_index = random.choice([x for x in range(len(self.video_list))])

            negetive_name = self.video_list[negetive_index]
            negetive_a = negetive_name.split('#')[0]
            while (negetive_a==a):
                negetive_index = random.choice([x for x in range(len(self.video_list))])
                negetive_name = self.video_list[negetive_index]
                negetive_a = negetive_name.split('#')[0]

            negetive_splits = negetive_name.split('#')
            if len(negetive_splits) == 2:
                negetive_a, negetive_b = negetive_splits
                negetive_npy_path = os.path.join(self.hdtf_dir, negetive_name+'.npy')
            else:
                negetive_a,negetive_b,negetive_c,negetive_d = negetive_splits
                negetive_npy_path = os.path.join(self.root_dir, negetive_a, '3DMM', negetive_b, negetive_c,negetive_d+'.npy')

            negetive_coeffs_pred_numpy = np.load(negetive_npy_path, allow_pickle=True)
            negetive_coeffs_pred_numpy = dict(enumerate(negetive_coeffs_pred_numpy.flatten(), 0))[0]
            negetive_coeff = negetive_coeffs_pred_numpy['coeff']
            negetive_coeff_mouth = negetive_coeff[:,80:144]
            
            
            while(len(negetive_coeff_mouth)<50):
                negetive_coeff_mouth = np.concatenate((negetive_coeff_mouth,negetive_coeff_mouth[::-1,:]),axis = 0)

            negetive_len_ = len(negetive_coeff_mouth)
            negetive_r = random.choice([x for x in range(3,negetive_len_-32)])
            negetive_coeff_mouth = negetive_coeff_mouth[negetive_r:negetive_r+32,:]

            negetive_parameters = np.array(negetive_coeff_mouth)
        return parameters, label, style_parameters, positive_parameters, negetive_parameters

