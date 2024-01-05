import argparse
import os
import data as Dataset
from config import Config
from logging import init_logging, make_logging_dir
from trainer_dis import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from distributed import init_dist
from distributed import master_only_print as print
import torch.nn as nn

from FaceRender_with_facial_discriminator.generators.FacialComponentDiscriminator import *

print('loaded pakeage2!')

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='config/facerender_discriminator.yaml')
    parser.add_argument('--name', default='v2')
    parser.add_argument('--checkpoints_dir', default='PIRender_with_facialdiscriminator',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=470000)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


def main():
    # get training options
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=True)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = opt.local_rank

    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    print(logdir)
    make_logging_dir(logdir, date_uid)
    # create a dataset
    val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)

    print("datasets loaded")

    # create a model
    net_G, net_G_ema, opt_G, sch_G \
        = get_model_optimizer_and_scheduler(opt)

    cri_component = build_loss(opt['gan_component_opt']).to('cuda')
    cri_l1 = build_loss(opt['L1_opt']).to('cuda')
    cri_gan = build_loss(opt['gan_opt']).to('cuda')

    optim_type = opt['optim_component'].pop('type')
    lr = opt['optim_component']['lr']


    net_d_left_eye = FacialComponentDiscriminator()
    net_d_right_eye = FacialComponentDiscriminator()
    net_d_mouth = FacialComponentDiscriminator()
    # left eye
    opt_d_left_eye = get_optimizer_d(
        optim_type,net_d_left_eye.parameters(), lr, betas=(0.9, 0.99))
    opt_d_right_eye = get_optimizer_d(
        optim_type,net_d_right_eye.parameters(), lr, betas=(0.9, 0.99))
    opt_d_mouth = get_optimizer_d(
        optim_type,net_d_mouth.parameters(), lr, betas=(0.9, 0.99))
    
    pre_train_discriminator_path = 'epoch_00225_iteration_000470000_checkpoint.pt'
    if pre_train_discriminator_path:
        pre_checkpoint_left = torch.load(pre_train_discriminator_path)
        net_d_left_eye.load_state_dict(pre_checkpoint_left['net_d_left_eye'])
        opt_d_left_eye.load_state_dict(pre_checkpoint_left['opt_d_left_eye'])
        net_d_right_eye.load_state_dict(pre_checkpoint_left['net_d_right_eye'])
        opt_d_right_eye.load_state_dict(pre_checkpoint_left['opt_d_right_eye'])
        net_d_mouth.load_state_dict(pre_checkpoint_left['net_d_mouth'])
        opt_d_mouth.load_state_dict(pre_checkpoint_left['opt_d_mouth'])


    if opt.distributed:
        net_d_left_eye = nn.parallel.DistributedDataParallel(
            net_d_left_eye.cuda(),
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        net_d_right_eye = nn.parallel.DistributedDataParallel(
            net_d_right_eye.cuda(),
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        net_d_mouth = nn.parallel.DistributedDataParallel(
            net_d_mouth.cuda(),
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    trainer = get_trainer(opt, net_G, net_d_left_eye, net_d_right_eye, net_d_mouth, net_G_ema, opt_G, sch_G, opt_d_left_eye, opt_d_right_eye, opt_d_mouth, train_dataset)

    

    current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter)   
    # # training flag
    max_epoch = opt.max_epoch

    if args.debug:
        trainer.test_everything(train_dataset, val_dataset, current_epoch, current_iteration)
        exit()
    # # Start training.
    for epoch in range(current_epoch, opt.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_dataset.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        
        for it, data in enumerate(train_dataset):
            data = trainer.start_of_iteration(data, current_iteration)
            # optimize facial component discriminators
            trainer.optimize_parameters_d(data)
            trainer.optimize_parameters(data)
            current_iteration += 1
            trainer.end_of_iteration(data, current_epoch, current_iteration)
 
            if current_iteration >= opt.max_iter:
                print('Done with training!!!')
                break
        current_epoch += 1
        trainer.end_of_epoch(data, val_dataset, current_epoch, current_iteration)

import time
from interval import Interval

if __name__ == '__main__':
    main()
    while True:
        # 当前时间
        now_localtime = time.strftime("%H:%M:%S", time.localtime())
        # 当前时间（以时间区间的方式表示）
        now_time = Interval(now_localtime, now_localtime)
    
        time_interval = Interval("23:30:00", "23:50:00")
    
        if now_time in time_interval:
            print("是在这个时间区间内")
            print("要执行的代码部分")
            main()
