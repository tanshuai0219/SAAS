import cv2
import json
import numpy as np
import os
import torch
from basicsr.utils import FileClient, imfrombytes
from collections import OrderedDict

# ---------------------------- This script is used to parse facial landmarks ------------------------------------- #
# Configurations
save_img = False
scale = 1  # 0.5 for official FFHQ (512x512), 1 for others
enlarge_ratio = 1.4  # only for eyes


# print('Load JSON metadata...')
# # use the official json file in FFHQ dataset
# with open(json_path, 'rb') as f:
#     json_data = json.load(f, object_pairs_hook=OrderedDict)

# print('Open LMDB file...')
# read ffhq images
# file_client = FileClient('lmdb', db_paths=face_path)
# with open(os.path.join(face_path, 'meta_info.txt')) as fin:
#     paths = [line.split('.')[0] for line in fin]

save_dict = {}

videos = []
with open('lists/train.json',"r") as f:
    videos += json.load(f)
with open('lists/test.json',"r") as f:
    videos += json.load(f)
print(len(videos))

hdtf_save_dir = 'datasets/HDTF'
mead_save_dir = 'datasets/processed_MEAD_front'

for item_idx, video in enumerate(videos):
    print(f'\r{item_idx} / {len(videos)}, {video} ', end='', flush=True)

    # parse landmarks
    splits = video.split('#')
    if len(splits)==2:
        lms = np.load(os.path.join(hdtf_save_dir,'landmark', video+'.npy'))
        save_path = os.path.join(hdtf_save_dir,'eye_mouth_landmarks', video+'.npy')
    else:
        a,b,c,d = splits
        dir2 = os.path.join(mead_save_dir, a, 'ldmk',b,c)
        video_name2 = os.path.join(dir2, d+'.npy')
        lms = np.load(video_name2)

        dir = os.path.join(mead_save_dir, a, 'eye_mouth_landmarks',b,c)
        os.makedirs(dir, exist_ok=True)
        save_path = os.path.join(dir, d+'.pth')

    item_dict = {}
    mouth_dict = []
    left_eye_dict = []
    right_eye_dict = []

    # get image

    # get landmarks for each component
    map_left_eye = list(range(36, 42))
    map_right_eye = list(range(42, 48))
    map_mouth = list(range(48, 68))
    for lm in lms:
        # eye_left
        mean_left_eye = np.mean(lm[map_left_eye], 0)  # (x, y)
        half_len_left_eye = np.max((np.max(np.max(lm[map_left_eye], 0) - np.min(lm[map_left_eye], 0)) / 2, 16))
        left_eye_dict.append([mean_left_eye[0], mean_left_eye[1], half_len_left_eye])
        
        # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
        half_len_left_eye *= enlarge_ratio
        loc_left_eye = np.hstack((mean_left_eye - half_len_left_eye + 1, mean_left_eye + half_len_left_eye)).astype(int)

        # eye_right
        mean_right_eye = np.mean(lm[map_right_eye], 0)
        half_len_right_eye = np.max((np.max(np.max(lm[map_right_eye], 0) - np.min(lm[map_right_eye], 0)) / 2, 16))
        right_eye_dict.append([mean_right_eye[0], mean_right_eye[1], half_len_right_eye])
        # mean_right_eye[0] = 512 - mean_right_eye[0]  # # for testing flip
        half_len_right_eye *= enlarge_ratio
        loc_right_eye = np.hstack(
            (mean_right_eye - half_len_right_eye + 1, mean_right_eye + half_len_right_eye)).astype(int)
        # mouth
        mean_mouth = np.mean(lm[map_mouth], 0)
        half_len_mouth = np.max((np.max(np.max(lm[map_mouth], 0) - np.min(lm[map_mouth], 0)) / 2, 16))
        mouth_dict.append([mean_mouth[0], mean_mouth[1], half_len_mouth])
        # mean_mouth[0] = 512 - mean_mouth[0]  # for testing flip
        loc_mouth = np.hstack((mean_mouth - half_len_mouth + 1, mean_mouth + half_len_mouth)).astype(int)
    item_dict['mouth'] = mouth_dict
    item_dict['right_eye'] = right_eye_dict
    item_dict['left_eye'] = left_eye_dict
    torch.save(item_dict, save_path)


# print('Save...')
# torch.save(save_dict, save_path)
