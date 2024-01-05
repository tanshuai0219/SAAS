import argparse, sys
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
from datasets.audio2pose_dataset import StyleDataset_audio_driven_pose

from style_extraction.models.vqgan.vqmodules.gan_models import setup_style_encoder, setup_vq_transformer

def gather_data(config, parameters, audio_feature, l_vq_model,
                                                    style_encoder, patch_size, seq_len):

    speakerData_np = parameters[:,:seq_len,80:144]
    listenerData_np = torch.cat([parameters[:,:,224:227], parameters[:,:,254:257]],2)#Y[idxStart:(idxStart + config['batch_size']), :, :]
    audioData_np = audio_feature

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
        parameters, _,_,audio_feature = bi
        parameters = parameters.cuda()
        audio_feature = audio_feature.cuda()
        inputs, listener_future, _, _ = gather_data(config, parameters, audio_feature, l_vq_model,
                                                    style_encoder, patch_size, seq_len)


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
        parameters, _,_,audio_feature = bi
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
        # if epoch%config['save_epoch'] == 0:
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
    
    style_path = 'checkpoints/style_extraction/style_extraction.pth'
    checkpoints = torch.load(style_path)
    
    style_encoder.load_state_dict( multi2single(checkpoints['state_dict']), strict=False)


    style_encoder.eval()
    # set up Predictor model
    fileName = config['model_path'] + \
                    '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None

    generator, g_optimizer, start_epoch = setup_model(config, l_vqconfig,
                                                      s_vqconfig=None,
                                                      load_path=None)
    generator.train()



    dataset = StyleDataset_audio_driven_pose(mead = 'processed_MEAD_front', hdtf = 'HDTF_5s', is_train=True)
    test_dataset = StyleDataset_audio_driven_pose(mead = 'processed_MEAD_front', hdtf = 'HDTF_5s', is_train=False)

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    start_epoch = start_epoch

    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch+config['num_epochs']-1:
            print('early stopping at:', epoch)
            print('best loss:', currBestLoss)
            break

        generator_train_step(config, epoch, generator, g_optimizer, l_vq_model,style_encoder,
                            dataloader, rng, writer,
                            patch_size, seq_len)


        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, generator, g_optimizer, l_vq_model, style_encoder, test_dataloader, 
            currBestLoss, prev_save_epoch, tag, writer, patch_size, seq_len)

    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/audio2pose/audio2pose.json')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    main(args)
