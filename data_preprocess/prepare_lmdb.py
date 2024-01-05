import os
import cv2
import lmdb
import argparse
import multiprocessing
import numpy as np

from glob import glob
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
from torchvision.transforms import functional as trans_fn

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class Resizer:
    def __init__(self, size, kp_root, coeff_3dmm_root, img_format):
        self.size = size
        self.kp_root = kp_root
        self.coeff_3dmm_root = coeff_3dmm_root
        self.img_format = img_format

    def get_resized_bytes(self, img, img_format='jpeg'):
        img = trans_fn.resize(img, (self.size, self.size), interpolation=Image.BICUBIC)
        buf = BytesIO()
        img.save(buf, format=img_format)
        img_bytes = buf.getvalue()
        return img_bytes

    def prepare(self, filename):
        frames = {'img':[], 'kp':None, 'coeff_3dmm':None}
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame)
                img_bytes = self.get_resized_bytes(img_pil, self.img_format)
                frames['img'].append(img_bytes)
            else:
                break
        cap.release()
        video_name = os.path.splitext(os.path.basename(filename))[0]
        keypoint_byte = get_others(self.kp_root, video_name, 'keypoint')
        coeff_3dmm_byte = get_others(self.coeff_3dmm_root, video_name, 'coeff_3dmm')
        frames['kp'] = keypoint_byte
        frames['coeff_3dmm'] = coeff_3dmm_byte
        return frames

    def __call__(self, index_filename):
        index, filename = index_filename
        result = self.prepare(filename)
        return index, result, filename

def get_others(root, video_name, data_type):
    if root is None:
        return
    else:
        assert data_type in ('keypoint', 'coeff_3dmm')
    if os.path.isfile(os.path.join(root, 'train', video_name+'.mat')):
        file_path = os.path.join(root, 'train', video_name+'.mat')
    else:
        file_path = os.path.join(root, 'test', video_name+'.mat')
    
    if data_type == 'keypoint':
        return_byte = convert_kp(file_path)
    else:
        return_byte = convert_3dmm(file_path)
    return return_byte

def convert_kp(file_path):
    file_mat = loadmat(file_path)
    kp_byte = file_mat['landmark'].tobytes()
    return kp_byte

def convert_3dmm(file_path):
    file_mat = np.load(file_path, allow_pickle=True)
    file_mat = dict(enumerate(file_mat.flatten(), 0))[0]
    coeff_3dmm = file_mat['coeff']
    crop_param = file_mat['transform_params']
    _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
    crop_param = np.concatenate([ratio, t0, t1], 1)
    coeff_3dmm_cat = np.concatenate([coeff_3dmm, crop_param], 1) 
    coeff_3dmm_byte = coeff_3dmm_cat.tobytes()
    return coeff_3dmm_byte

def convert_kp_3dmm(file_path):
    file_mat = np.load(file_path, allow_pickle=True)
    file_mat = dict(enumerate(file_mat.flatten(), 0))[0]
    coeff_3dmm = file_mat['coeff']
    crop_param = file_mat['trans_param']
    _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
    crop_param = np.concatenate([ratio, t0, t1], 1)
    coeff_3dmm_cat = np.concatenate([coeff_3dmm, crop_param], 1) 
    coeff_3dmm_byte = coeff_3dmm_cat.tobytes()
    kp_byte = file_mat['lm68'].tobytes()
    return kp_byte, coeff_3dmm_byte

def convert_coeff_3dmm(file_path):
    file_mat = np.load(file_path, allow_pickle=True)
    file_mat = dict(enumerate(file_mat.flatten(), 0))[0]
    coeff_3dmm = file_mat['coeff']
    coeff_3dmm_byte = coeff_3dmm.tobytes()
    return coeff_3dmm_byte

def convert_mel(audio_file):
    audio_feature = np.load(audio_file, allow_pickle=True)
    len_ = int(audio_feature.shape[0])
    audio_feature = audio_feature.tobytes()
    return audio_feature, len_

def get_others_HDTF(root, video_name):

    file_path = os.path.join(root, '3DMM', video_name+'.npy')
    audio_file = os.path.join(root, 'mel', video_name+'.npy')
    coeff_3dmm_byte = convert_coeff_3dmm(file_path)
    mel_byte, len_ = convert_mel(audio_file)


    return mel_byte, coeff_3dmm_byte,len_

def get_others_MEAD(root, video_name):
    a,b,c,d = video_name.split('#')
    file_path = os.path.join(root, a, '3DMM',b,c, d+'.npy')
    audio_path = os.path.join(root, a, 'mel',b,c, d+'.npy')
    
    coeff_3dmm_byte = convert_coeff_3dmm(file_path)
    mel_byte, len_ = convert_mel(audio_path)


    return mel_byte, coeff_3dmm_byte,len_

class Resizer_MEAD_HDTF:
    def __init__(self, MEAD_path, HDTF_path):
        self.MEAD_path = MEAD_path
        self.HDTF_path = HDTF_path
        self.img_format = 'jpeg'
        self.size = 256

    def prepare_HDTF(self, filename):
        # print(filename)
        frames = {'mel':None, 'coeff_3dmm':None}

        mel_byte, coeff_3dmm_byte,len_ = get_others_HDTF(self.HDTF_path, filename)
        frames['coeff_3dmm'] = coeff_3dmm_byte
        frames['mel'] = mel_byte
        frames['len'] = len_
        return frames

    def prepare_MEAD(self, filename):
        # print(filename)
        frames = {'mel':None, 'coeff_3dmm':None}
        a,b,c,d = filename.split('#')


        mel_byte, coeff_3dmm_byte,len_ = get_others_MEAD(self.MEAD_path, filename)
        frames['coeff_3dmm'] = coeff_3dmm_byte
        frames['mel'] = mel_byte
        frames['len'] = len_
        return frames

    def get_resized_bytes(self, img, img_format='jpeg'):
        img = trans_fn.resize(img, (self.size, self.size), interpolation=Image.BICUBIC)
        buf = BytesIO()
        img.save(buf, format=img_format)
        img_bytes = buf.getvalue()
        return img_bytes

    def __call__(self, index_filename):
        index, filename = index_filename
        if len(filename.split('#')) == 2:
            result = self.prepare_HDTF(filename)
        else:
            result = self.prepare_MEAD(filename)
        return index, result, filename



import json
def prepare_data(MEAD_path,HDTF_path, out, n_worker, chunksize):
    
    train_file = 'lists/SAAS/train.json'
    with open(train_file,"r") as f:
        train_video = json.load(f)
    test_file = 'lists/SAAS/test.json'
    with open(test_file,"r") as f:
        test_video = json.load(f)
    
    filenames = train_video+test_video

    new_train = []
    new_test = []

    filenames = sorted(filenames)
    total = len(filenames)
    os.makedirs(out, exist_ok=True)
    lmdb_path = out
    with lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            txn.put(format_for_lmdb('length'), format_for_lmdb(total))
            resizer = Resizer_MEAD_HDTF(MEAD_path,HDTF_path)
            with multiprocessing.Pool(n_worker) as pool:
                for idx, result, filename in tqdm(
                        pool.imap_unordered(resizer, enumerate(filenames), chunksize=chunksize),
                        total=total):
                    filename = os.path.basename(filename)
                    video_name = os.path.splitext(filename)[0]
                    # print(video_name)
                    if result['len']<32:
                        continue

                    if video_name in train_video:
                        new_train.append(video_name)
                    elif video_name in test_video:
                        new_test.append(video_name)
                    txn.put(format_for_lmdb(video_name, 'length'), format_for_lmdb(result['len']))

                    if result['coeff_3dmm']:
                        txn.put(format_for_lmdb(video_name, 'coeff_3dmm'), result['coeff_3dmm'])
                    if result['mel']:
                        txn.put(format_for_lmdb(video_name, 'mel'), result['mel'])
    label_file = 'lists/train.json'
    with open(label_file,'w') as file_obj:
        json.dump(new_train,file_obj)
    label_file = 'lists/test.json'
    with open(label_file,'w') as file_obj:
        json.dump(new_test,file_obj)  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--MEAD_path', type=str, help='a path to input directiory', default=None)
    parser.add_argument('--HDTF_path', type=str, help='a path to input directiory', default=None)
    parser.add_argument('--out', type=str, help='a path to output directory', default = 'SAAS_lmdb')
    parser.add_argument('--n_worker', type=int, help='number of worker processes', default=8)
    parser.add_argument('--chunksize', type=int, help='approximate chunksize for each worker', default=10)
    args = parser.parse_args()
    prepare_data(**vars(args))
