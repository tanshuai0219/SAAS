import sys, os
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
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
from datasets.style_extraction_dataset import StyleDataset_four
from style_extraction.models.vqgan.vqmodules.gan_models import setup_vq_transformer, calc_vq_loss
import sys
from style_extraction.models.utils.load_utils import *
import torch.nn.functional as F
from distributed import init_dist
from distributed import master_only_print as print

def generator_train_step(config, epoch, generator, g_optimizer, train_dataloader, writer):

    generator.train()
    criterion = nn.CrossEntropyLoss()
    totalSteps = len(train_dataloader)
    avgLoss = 0
    crossLoss = 0
    TripLoss = 0
    t_argloss = 0

    acc_total = 0

    for bii, bi in enumerate(train_dataloader):
        gtData, labels, style_parameters, positive_parameters, negetive_parameters = bi
        gtData = gtData.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()
        positive_parameters = positive_parameters.type(torch.FloatTensor).cuda()
        negetive_parameters = negetive_parameters.type(torch.FloatTensor).cuda()

        prediction, quant_loss, style, style_code, style_style_code, positive_style_code, negetive_style_code = generator(gtData, style_parameters, positive_parameters, negetive_parameters)
        g_loss = calc_vq_loss(prediction, gtData.cuda(), quant_loss)
        cross_loss = criterion(style, labels)

        trip_loss = max(2*F.mse_loss(style_code, positive_style_code)-F.mse_loss(style_code, style_style_code)-F.mse_loss(style_code, negetive_style_code)+5,0)
        aaa = 5
        if type(trip_loss) == type(aaa):
            total_loss = g_loss*config['loss_weights']['g_loss'] \
                + cross_loss*config['loss_weights']['cross_loss'] \
                    + trip_loss*config['loss_weights']['trip_loss']
        else:

            total_loss = g_loss*config['loss_weights']['g_loss'] \
                    + cross_loss*config['loss_weights']['cross_loss']

        _, predicted = torch.max(style.data, 1)
        acc = ((predicted == labels).sum().item())/((labels == labels).sum().item())

        g_optimizer.zero_grad()

        total_loss.backward()

        g_optimizer.step_and_update_lr()
        avgLoss += g_loss.detach().item()
        crossLoss += cross_loss.detach().item()
        try:
            TripLoss += trip_loss.detach().item()
        except:
            TripLoss = trip_loss
        t_argloss += total_loss.detach().item()
        acc_total += acc

        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], G_Loss: {:.4f}, acc:{:.4f},Cross_Loss: {:.4f}, Trip_loss:{:.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            g_loss.detach().item(),acc, cross_loss.detach().item(), trip_loss.detach().item()))

    writer.add_scalar('Loss/train_totalLoss', t_argloss / totalSteps, epoch)
    writer.add_scalar('acc', acc_total / totalSteps, epoch)
    writer.add_scalar('g_loss/train_totalLoss', avgLoss / totalSteps, epoch)
    writer.add_scalar('crossLoss/train_crossLoss', crossLoss / totalSteps, epoch)
    writer.add_scalar('Trip/train_crossLoss', TripLoss / totalSteps, epoch)


def generator_val_step(config, epoch, generator, g_optimizer, test_dataloader,
                       currBestLoss,currBestAcc, prev_save_epoch, tag, writer):
    """ Function that validates training of VQ-VAE

    see generator_train_step() for parameter definitions
    """
    os.makedirs(config['model_path'], exist_ok=True)
    generator.eval()
    criterion = nn.CrossEntropyLoss()
    totalSteps = len(test_dataloader)
    

    t_avgLoss = 0
    g_avgLoss = 0
    cross_avgLoss = 0
    trip_avgLoss = 0
    acc_total = 0

    for bii, bi in enumerate(test_dataloader):
        gtData, labels, style_parameters, positive_parameters, negetive_parameters = bi
        gtData = gtData.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()
        positive_parameters = positive_parameters.type(torch.FloatTensor).cuda()
        negetive_parameters = negetive_parameters.type(torch.FloatTensor).cuda()
        gtData = gtData.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()

        with torch.no_grad():
            prediction, quant_loss, style, style_code, style_style_code, positive_style_code, negetive_style_code = generator(gtData, style_parameters, positive_parameters, negetive_parameters)
        g_loss = calc_vq_loss(prediction, gtData.cuda(), quant_loss)
        cross_loss = criterion(style, labels)

        trip_loss = max(2*F.mse_loss(style_code, positive_style_code)-F.mse_loss(style_code, style_style_code)-F.mse_loss(style_code, negetive_style_code)+5,0)

        total_loss = g_loss*config['loss_weights']['g_loss'] \
        + cross_loss*config['loss_weights']['cross_loss'] \
            + trip_loss*config['loss_weights']['trip_loss']

        t_avgLoss += total_loss.detach().item()
        g_avgLoss += g_loss.detach().item()
        cross_avgLoss += cross_loss.detach().item()
        trip_avgLoss += trip_loss.detach().item()
        try:
            trip_avgLoss += trip_loss.detach().item()
        except:
            trip_avgLoss += trip_loss
        _, predicted = torch.max(style.data, 1)
        acc = ((predicted == labels).sum().item())/((labels == labels).sum().item())
        acc_total += acc
        if bii % (config['log_step']/2) == 0:
            print('val_Epoch [{}/{}], Step [{}/{}], G_Loss: {:.4f}, acc:{:.4f},Cross_Loss: {:.4f}, Trip_loss:{:.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            g_loss.detach().item(),acc, cross_loss.detach().item(), trip_loss.detach().item()))

    t_avgLoss /= totalSteps
    acc_total /= totalSteps

    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', t_avgLoss / totalSteps, epoch)
    writer.add_scalar('acc/val_totalLoss', acc_total, epoch)
    writer.add_scalar('g_loss/val_totalLoss', g_avgLoss / totalSteps, epoch)
    writer.add_scalar('crossLoss/val_crossLoss', cross_avgLoss / totalSteps, epoch)
    writer.add_scalar('Trip/val_crossLoss', trip_avgLoss / totalSteps, epoch)
    ## save model if curr loss is lower than previous best loss
    if t_avgLoss < currBestLoss or acc_total > currBestAcc:
        prev_save_epoch = epoch
        checkpoint = {'config': args.config,
                      'state_dict': generator.state_dict(),
                      'optimizer': {
                        'optimizer': g_optimizer._optimizer.state_dict(),
                        'n_steps': g_optimizer.n_steps,
                      },
                      'epoch': epoch}
        if t_avgLoss < currBestLoss:
            fileName = config['model_path'] + \
                            '{}{}_best_{}.pth'.format(tag, config['pipeline'], '%04d'%epoch)
            currBestLoss = t_avgLoss
            print('>>>> saving best epoch {}'.format(epoch), t_avgLoss)
        else:
            currBestAcc = acc_total
            fileName = config['model_path'] + \
                                        '{}{}_acc_{}.pth'.format(tag, config['pipeline'], '%04d'%epoch)
            print('>>>> saving best epoch {} acc {}'.format(epoch, acc_total), t_avgLoss)
        
        torch.save(checkpoint, fileName)
        return currBestLoss, currBestAcc, prev_save_epoch, t_avgLoss

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
                            '{}{}_no_best_{}.pth'.format(tag, config['pipeline'], '%04d'%epoch)
            
            torch.save(checkpoint, fileName)
        return currBestLoss,currBestAcc, prev_save_epoch, currBestLoss

from collections import OrderedDict
def multi2single(pretrain):
    new_state_dict = OrderedDict()
    for key, value in pretrain.items():
        name = key[7:]
        new_state_dict[name] = value
    return new_state_dict

def main(args):
    """ full pipeline for training the Predictor model """

    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
    tag = config['tag']
    pipeline = config['pipeline']
    currBestLoss = 1e3
    currBestAcc = 0
    ## can modify via configs, these are default for released model

    prev_save_epoch = 0
    writer = SummaryWriter('runs/debug_{}_{}'.format(tag, pipeline))

    ## setting up models
    fileName = config['model_path'] + \
                '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None

    

    generator, g_optimizer, start_epoch = setup_vq_transformer(args, config,
                                            version=None, load_path=None)
    generator.train()
    loaded_state = torch.load(load_path)
    generator.load_state_dict(multi2single(loaded_state['state_dict']), strict=True)
    start_epoch = loaded_state['epoch']+1
    g_optimizer._optimizer.load_state_dict(
                                loaded_state['optimizer']['optimizer'])

    generator=torch.nn.parallel.DistributedDataParallel(generator.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)

    train_dataset = StyleDataset_four(root_dir = 'processed_MEAD_front',hdtf_dir = 'HDTF/split_5s_3DMM', is_train=True)
    test_dataset = StyleDataset_four(root_dir = 'processed_MEAD_front',hdtf_dir = 'HDTF/split_5s_3DMM', is_train=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None), num_workers=16, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=(test_sampler is None), num_workers=16, sampler=test_sampler, pin_memory=True, drop_last=True)

    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch+config['num_epochs']-1:
            print('early stopping at:', epoch)
            print('best loss:', currBestLoss)
            break
        train_dataloader.sampler.set_epoch(epoch)
        test_dataloader.sampler.set_epoch(epoch)
        generator_train_step(config, epoch, generator, g_optimizer, train_dataloader, writer)

        currBestLoss,currBestAcc, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, generator, g_optimizer, test_dataloader,
                               currBestLoss,currBestAcc, prev_save_epoch, tag, writer)
    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/style_extraction/style_extraction.json')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    init_dist(args.local_rank)
    main(args)
