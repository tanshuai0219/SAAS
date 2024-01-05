import sys, os
sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')
import argparse
import json
import logging
import numpy as np
import os
import scipy.io as sio
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datasets.audio2pose_dataset import StyleDataset_pose_mead_hdtf
from audio2pose.models.vqgan.vqmodules.gan_models import setup_vq_transformer, calc_vq_loss_pose
from models.utils.load_utils import *
import torch.nn.functional as F

def generator_train_step(config, epoch, generator, g_optimizer, train_dataloader, writer):
    """ Function to do autoencoding training for VQ-VAE

    Parameters
    ----------
    generator:
        VQ-VAE model that takes as input continuous listener and learns to
        outputs discretized listeners
    g_optimizer:
        optimizer that trains the VQ-VAE
    train_X:
        continuous listener motion sequence (acts as the target)
    """

    generator.train()
    totalSteps = len(train_dataloader)
    avgLoss = avgDLoss = 0
    crossLoss = 0
    for bii, bi in enumerate(train_dataloader):
        gtData = bi
        gtData = gtData.type(torch.FloatTensor).cuda()

        prediction, quant_loss, style, _ = generator(gtData, None)
        g_loss = calc_vq_loss_pose(prediction, gtData.cuda(), quant_loss)
        total_loss = g_loss
        g_optimizer.zero_grad()
        total_loss.backward()
        g_optimizer.step_and_update_lr()
        avgLoss += g_loss.detach().item()
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            total_loss.detach().item(), np.exp(total_loss.detach().item())))
    writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)


def generator_val_step(config, epoch, generator, g_optimizer, test_dataloader,
                       currBestLoss, prev_save_epoch, tag, writer):
    """ Function that validates training of VQ-VAE

    see generator_train_step() for parameter definitions
    """

    generator.eval()
    totalSteps = len(test_dataloader)
    testLoss = testDLoss = 0
    for bii, bi in enumerate(test_dataloader):
        gtData = bi
        gtData = gtData.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            prediction, quant_loss, style, _ = generator(gtData, None)
        g_loss = calc_vq_loss_pose(prediction, gtData, quant_loss)
        testLoss += (g_loss).detach().item()
    testLoss /= totalSteps
    print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                .format(epoch, config['num_epochs'], bii, totalSteps,
                        testLoss, np.exp(testLoss)))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss / totalSteps, epoch)

    ## save model if curr loss is lower than previous best loss
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
                        '{}{}_best_{}.pth'.format(tag, config['pipeline'], '%04d'%epoch)
        print('>>>> saving best epoch {}'.format(epoch), testLoss)
        currBestLoss = testLoss
        torch.save(checkpoint, fileName)
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
            os.makedirs(config['model_path'], exist_ok=True)
            fileName = config['model_path'] + \
                            '{}{}_no_best_{}.pth'.format(tag, config['pipeline'], '%04d'%epoch)
            
            torch.save(checkpoint, fileName)
        return currBestLoss, prev_save_epoch, currBestLoss


def main(args):
    """ full pipeline for training the Predictor model """

    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
    tag = config['tag']
    pipeline = config['pipeline']
    currBestLoss = 1e3
    ## can modify via configs, these are default for released model

    prev_save_epoch = 0
    writer = SummaryWriter('runs/debug_{}{}'.format(tag, pipeline))

    ## setting up models
    fileName = config['model_path'] + \
                '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None
    generator, g_optimizer, start_epoch = setup_vq_transformer(args, config,
                                            version=None, load_path=load_path)
    generator.train()


    train_dataset = StyleDataset_pose_mead_hdtf(root_dir = 'processed_MEAD',root_dir_HDTF='HDTF/split', is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    test_dataset = StyleDataset_pose_mead_hdtf(root_dir = 'processed_MEAD',root_dir_HDTF='HDTF/split', is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    disc_factor = 0.0
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch+config['num_epochs']-1:
            print('early stopping at:', epoch)
            print('best loss:', currBestLoss)
            break
        generator_train_step(config, epoch, generator, g_optimizer, train_dataloader, writer)

        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, generator, g_optimizer, test_dataloader,
                               currBestLoss, prev_save_epoch, tag, writer)
    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/audio2pose/pose_encoder.json')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    main(args)
