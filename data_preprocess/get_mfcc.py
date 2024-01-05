# -*- coding: utf-8 -*-

import os

import librosa
import numpy as np
import python_speech_features
from pathlib import Path


def audio2mfcc(audio_file, save):
    speech, sr = librosa.load(audio_file, sr=16000)

    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)
    if not os.path.exists(save):
        os.makedirs(save)
    time_len = mfcc.shape[0]

    for input_idx in range(int((time_len-28)/4)+1):

        input_feat = mfcc[4*input_idx:4*input_idx+28,:]

        np.save(os.path.join(save, '%03d.npy'%(input_idx)), input_feat)


    print(input_idx)


audio_file = ''
save_path = ''
audio2mfcc(audio_file, save_path)

