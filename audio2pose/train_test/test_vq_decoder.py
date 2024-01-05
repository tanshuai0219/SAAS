import argparse, sys
import librosa
import python_speech_features

sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
import json
import logging
import numpy as np
import os

import scipy.io as sio
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

from audio2pose.models.modules.fact_model import setup_model, calc_logit_loss
# from vqgan.vqmodules.gan_models import setup_vq_transformer
from audio2pose.models.utils.base_model_util import *
from audio2pose.models.utils.load_utils import *
# from data_process.process_audio_hubert import generate_hubert

from style_extraction.models.vqgan.vqmodules.gan_models import setup_style_encoder, setup_vq_transformer


def gather_data(config, parameters, audio_feature, l_vq_model,
                                                    style_encoder, patch_size, seq_len):
    """ method to prepare data into proper format for training

    Parameters
    ----------
    X: tensor (B,T1,F)
        Past+current raw speaker motion of sequence length T1
    Y: tensor (B,T2,F)
        Past raw listener motion of sequence length T2
    audio: tensor (B,T3,A)
        Past raw speaker audio of sequence length T3
    l_vq_model:
        pre-trained VQ-VAE model used to discretize the past listener motion and
        decode future listener motion predictions
    patch_size: int
        patch length that we divide seq_len into for the VQ-VAE model
    seq_len: int
        full length of sequence that is taken as input into the VQ-VAE model
    bi: int
        current batch index
    """
    speakerData_np = parameters[:,seq_len:,80:144]
    listenerData_np = torch.cat([parameters[:,:,224:227], parameters[:,:,254:257]],2)#Y[idxStart:(idxStart + config['batch_size']), :, :]
    audioData_np = audio_feature
    # inputs = {"speaker_full": speaker_full,
    #           "listener_past": listener_past_index,
    #           "audio_full": audio_full}
    speakerData_np = style_encoder(speakerData_np)
    inputs, listener_future, raw_listener, btc = \
        create_data_vq(l_vq_model, speakerData_np, listenerData_np,
                        audioData_np, seq_len,
                        data_type=config['loss_config']['loss_type'],
                        patch_size=patch_size)
    return inputs, listener_future, raw_listener, btc


def generator_train_step(config, epoch, generator, g_optimizer, l_vq_model,
                         style_encoder, dataloader, rng, writer,
                         patch_size, seq_len):
    """ method to prepare data into proper format for training

    see gather_data() for remaining parameter definitions

    Parameters
    ----------
    epoch: int
    generator:
        Predictor model that outputs future listener motion conditioned on past
        listener motion and speaker past+current audio+motion
    g_optimizer:
        optimizer for training the Predictor model
    """

    generator.train()
    totalSteps = len(dataloader)
    avgLoss = 0

    for bii, bi in enumerate(dataloader):
        parameters, _, _,_,audio_feature = bi
        parameters = parameters.cuda()
        audio_feature = audio_feature.cuda()
        inputs, listener_future, _, _ = gather_data(config, parameters, audio_feature, l_vq_model,
                                                    style_encoder, patch_size, seq_len)
    # inputs = {"speaker_full": speaker_full,
    #           "listener_past": listener_past_index,
    #           "audio_full": audio_full}

        prediction = generator(inputs,
                        config['fact_model']['cross_modal_model']['max_mask_len'],
                        -1)
        cut_point = listener_future.shape[1]
        # print(listener_future.shape, prediction.shape)
        logit_loss = calc_logit_loss(prediction[:,:cut_point,:],
                                     listener_future[:,:cut_point])
        g_loss = logit_loss
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step_and_update_lr()
        avgLoss += g_loss.detach().item()
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            g_loss.detach().item(), np.exp(g_loss.detach().item())))
            avg_Loss = 0
    writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)


def generator_val_step(config, epoch, generator, g_optimizer, l_vq_model,
                       style_encoder, test_dataloader, currBestLoss,
                       prev_save_epoch, tag, writer, patch_size, seq_len):
    """ method to validate training of Predictor model

    see generator_train_step() for full parameters definition
    """

    generator.eval()
    
    totalSteps = len(test_dataloader)
    testLoss = 0

    for bii, bi in enumerate(test_dataloader):
        parameters, _, _,_,audio_feature = bi
        parameters = parameters.cuda()
        audio_feature = audio_feature.cuda()
        inputs, listener_future, _, _ = gather_data(config, parameters, audio_feature, l_vq_model,
                                                    style_encoder, patch_size, seq_len)
        with torch.no_grad():
            prediction = generator(inputs,
                config['fact_model']['cross_modal_model']['max_mask_len'], -1)
        cut_point = listener_future.shape[1]
        logit_loss = calc_logit_loss(prediction[:,:cut_point,:],
                                     listener_future[:,:cut_point])
        g_loss = logit_loss
        testLoss += g_loss.detach().item()

    testLoss /= totalSteps
    print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            testLoss, np.exp(testLoss)))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss, epoch)
    os.makedirs(config['model_path'], exist_ok=True)
    ## save model if the curent loss is better than previous best
    if testLoss < currBestLoss:
        prev_save_epoch = epoch
        checkpoint = {'config': args.config,
                      'state_dict': generator.state_dict(),
                      'optimizer': {
                        'optimizer': g_optimizer._optimizer.state_dict(),
                        'n_steps': g_optimizer.n_steps,
                      },
                      'epoch': epoch}
        fileName = config['model_path'] + \
                    '{}{}_best_{}.pth'.format(tag, config['pipeline'],'%04d'%epoch)
        currBestLoss = testLoss
        if epoch%config['save_epoch'] == 0:
            torch.save(checkpoint, fileName)
            print('>>>> saving best epoch {}'.format(epoch), testLoss)
        return currBestLoss, prev_save_epoch, testLoss
    else:
        if epoch%config['save_epoch'] == 0:
            prev_save_epoch = epoch
            checkpoint = {'config': args.config,
                        'state_dict': generator.state_dict(),
                        'optimizer': {
                            'optimizer': g_optimizer._optimizer.state_dict(),
                            'n_steps': g_optimizer.n_steps,
                        },
                        'epoch': epoch}
            fileName = config['model_path'] + \
                        '{}{}_no_best_{}.pth'.format(tag, config['pipeline'],'%04d'%epoch)
            torch.save(checkpoint, fileName)
            # print('>>>> saving best epoch {}'.format(epoch), testLoss)
        return currBestLoss, prev_save_epoch, testLoss


def generate_mfcc(audio_file):
    speech, sr = librosa.load(audio_file, sr=16000)
  #  mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)


    print ('=======================================')
    print ('Start to generate images')

    ind = 3
    with torch.no_grad():
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)

        

        input_mfcc = input_mfcc.unsqueeze(0)
    return input_mfcc


from scipy.signal import savgol_filter
def generate_one(config, l_vq_model, style_encoder, generator, target_name, root_dir, save_dir, tag):
    target_a, target_b, target_c = target_name.split('_')
    target_path = os.path.join(root_dir, target_a, 'video_25', target_b, target_c+'_coeff_pt.npy')
  
    target_coeffs_pred_numpy = np.load(target_path, allow_pickle=True)
    target_coeffs_pred_numpy = dict(enumerate(target_coeffs_pred_numpy.flatten(), 0))[0]
    target_coeff = target_coeffs_pred_numpy['coeff']
    target_coeff_mouth = target_coeff
    save_path = os.path.join(save_dir, target_name+'.npy')


    target_parameters = torch.from_numpy(np.array(target_coeff_mouth)).unsqueeze(0).cuda()

    target_audio_path = os.path.join(root_dir, target_a, 'audio_video', target_b, target_c+'.wav')
    target_audio_feature = generate_mfcc(target_audio_path).type(torch.FloatTensor).cuda()

    target_len = min(len(target_coeff_mouth), int(target_audio_feature.shape[1]))
    target_parameters = target_parameters[:,:target_len]
    target_audio_feature = target_audio_feature[:,:target_len]
    cur_len = target_len
    while (int(cur_len/32)*32 != cur_len):
        target_audio_feature = torch.cat([target_audio_feature, target_audio_feature[:,(cur_len-1):cur_len]],1)
        target_parameters = torch.cat([target_parameters, target_parameters[:,(cur_len-1):cur_len]],1)
        cur_len = target_parameters.shape[1]
    # print("--------1-----------")
    # print(target_audio_feature.shape[1],target_parameters.shape[1],)
    seq_len = 32
    speakerData_np = target_parameters[:,:seq_len,80:144]
    speakerData_np = style_encoder(speakerData_np)
    listenerData_np = torch.cat([target_parameters[:,:,224:227], target_parameters[:,:,254:257]],2)
    cut_point = config['fact_model']['listener_past_transformer_config']\
                    ['sequence_length']
    predictions = None

    inputs, _, raw_listener, quant_size = \
        create_data_vq_test(l_vq_model, speakerData_np, listenerData_np[:,:seq_len],
                        target_audio_feature[:,:seq_len], seq_len,
                        data_type=config['loss_config']['loss_type'])
    quant_prediction = generator(inputs,
                    config['fact_model']['cross_modal_model']['max_mask_len'],
                    0)

    prediction, probs = l_vq_model.get_logit( #prediction torch.Size([20, 1, 1])  probs  torch.Size([20, 1, 200])
                                        quant_prediction[:,:cut_point,:],
                                        sample_idx=None)
    prediction = torch.cat((inputs['listener_past'],
                                    prediction[:,0]), axis=-1)
    step_t = 8
    start_t = 0
    past_cut_point = 32       
    for t in range(start_t, target_parameters.shape[1]-past_cut_point, step_t):
        # print("--------2-----------")
        # print((t+step_t)*2,(t+(seq_len)+step_t)*2)
        listener_in = \
            prediction.data[:,int(t/step_t):int((t+seq_len)/step_t)]
        inputs, _, raw_listener, quant_size = \
            create_data_vq_test(l_vq_model, speakerData_np, listenerData_np[:,:seq_len],
                            target_audio_feature[:,(t+step_t):(t+(seq_len)+step_t)], seq_len,
                            data_type=config['loss_config']['loss_type'])
        inputs['listener_past'] = listener_in
        quant_prediction = generator(inputs,
                        config['fact_model']['cross_modal_model']['max_mask_len'],
                        int(t/step_t))

        if t == target_parameters.shape[1]-past_cut_point-step_t:
            for k in range(4):
                curr_prediction, probs = l_vq_model.get_logit( #prediction torch.Size([20, 1, 1])  probs  torch.Size([20, 1, 200])
                                                    quant_prediction[:,k:cut_point+k,:],
                                                    sample_idx=None)
                prediction = torch.cat((prediction, curr_prediction[:,0]), axis=1)
        else:
            curr_prediction, probs = l_vq_model.get_logit( #prediction torch.Size([20, 1, 1])  probs  torch.Size([20, 1, 200])
                                                quant_prediction[:,:cut_point,:],
                                                sample_idx=None)
            prediction = torch.cat((prediction, curr_prediction[:,0]), axis=1)

    decoded_pred = None

    prediction = prediction[:,quant_size[-1]:]
    for t in range(0, prediction.shape[-1]-3): # torch.Size([1, 8, 200])  (1,192,4)
        curr_decoded = l_vq_model.decode_to_img(
                            prediction[:,t:t+quant_size[-1]], quant_size)
        if t == prediction.shape[-1]-4:
            decoded_pred = curr_decoded if decoded_pred is None \
                else torch.cat((decoded_pred, curr_decoded), axis=1)
        else:
            decoded_pred = curr_decoded[:,:8] if decoded_pred is None \
                            else torch.cat((decoded_pred, curr_decoded[:,:8]), axis=1)
    #re-attach initial gt information (not used in eval)
    # prediction = torch.cat((listenerData_np[:,:seq_len,:].cuda(),
    #                                 decoded_pred), dim=1) # torch.Size([20, 64, 56])
    prediction = decoded_pred
    ## calculating upperbound of quantization by decoding and unencoding GT

    output_pred = savgol_filter(np.array(prediction.cpu()), 13, 2, axis=1)
    # output_pred = prediction.data.cpu().numpy()
    output_pred = output_pred[:,:target_len]
    # print('out', output_pred.shape)
    target_coeffs_pred_numpy['coeff'] = target_coeffs_pred_numpy['coeff'][:target_len]
    # print(target_coeffs_pred_numpy['coeff'][:, 224: 227].shape)
    # print(output_pred[0][:,224: 227].shape)
    target_coeffs_pred_numpy['coeff'][:, 80: 144] *= 0
    target_coeffs_pred_numpy['coeff'][:, 224: 227] = output_pred[0][:,: 3]
    target_coeffs_pred_numpy['coeff'][:, 254:] = output_pred[0][:,3:]
    np.save(save_path, target_coeffs_pred_numpy)
    print(save_path)
    return output_pred[:,:target_len]

from collections import OrderedDict
def multi2single(pretrain):
    new_state_dict = OrderedDict()
    for key, value in pretrain.items():
        name = key[7:]
        new_state_dict[name] = value
    return new_state_dict

def main(args):
    """ full pipeline for training the Predictor model """
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
    tag = config['tag']
    pipeline = config['pipeline']
    writer = SummaryWriter('runs/debug_{}{}'.format(tag, pipeline))
    args.get_attn = False
    currBestLoss = 1e3
    prev_save_epoch = 0
    ## can modify via configs, these are default for released model
    patch_size = 8
    seq_len = 32

    ## setting up the listener VQ-VAE and Predictor models
    # load pre-trained VQ-VAE model
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)
    l_model_path = 'checkpoints/audio2pose/pose_encoder.pth'
    l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path)
    for param in l_vq_model.parameters():
        param.requires_grad = False
    l_vq_model.eval()

    # load pre-trained style encoder model
    style_config = 'configs/style_extraction/style_exrtaction.json'
    with open(style_config) as f:
        style_vqconfig = json.load(f)
    style_encoder, g_optimizer, start_epoch = setup_style_encoder(args, style_vqconfig,
                                            version=None, load_path=None)
    for param in style_encoder.parameters():
        param.requires_grad = False
    style_encoder.eval()
    # set up Predictor model

    style_path = 'checkpoints/style_extraction/style_extraction.pth'
    checkpoints = torch.load(style_path)
    
    style_encoder.load_state_dict( multi2single(checkpoints['state_dict']), strict=False)


    style_encoder.eval()


    root_dir = ''
    save_dir = ''
    load_path = 'checkpoints/audio2pose/audio2pose.pth'
    generator, g_optimizer, start_epoch = setup_model(config, l_vqconfig,
                                                      s_vqconfig=None,
                                                        load_path=load_path)
    generator.eval()


    os.makedirs(save_dir, exist_ok=True)
    target_name = ''
    generate_one(config, l_vq_model, style_encoder, generator, target_name, root_dir, save_dir, tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/audio2pose/audio2pose.json')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    main(args)
