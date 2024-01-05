import math, sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
import random
import torch
print('loading basicsr....')
from basicsr.archs.stylegan2_arch import (ConvLayer, EqualConv2d, EqualLinear, ResBlock, ScaledLeakyReLU,
                                          StyleGAN2Generator)
from basicsr.ops.fused_act import FusedLeakyReLU
from basicsr.utils.registry import ARCH_REGISTRY
print('loaded')
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from copy import deepcopy
from config import Config
from torchvision.ops import roi_align
from collections import OrderedDict


class FacialComponentDiscriminator(nn.Module):
    """Facial component (eyes, mouth, noise) discriminator used in GFPGAN.
    """

    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()
        # It now uses a VGG-style architectrue with fixed model size
        self.conv1 = ConvLayer(3, 64, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv2 = ConvLayer(64, 128, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv3 = ConvLayer(128, 128, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv4 = ConvLayer(128, 256, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv5 = ConvLayer(256, 256, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False, **kwargs):
        """Forward function for FacialComponentDiscriminator.

        Args:
            x (Tensor): Input images.
            return_feats (bool): Whether to return intermediate features. Default: False.
        """
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)

        if return_feats:
            return out, rlt_feats
        else:
            return out, None

class AllFacialComponentDiscriminator(nn.Module):
    def __init__(self):
        super(AllFacialComponentDiscriminator, self).__init__()
        # It now uses a VGG-style architectrue with fixed model size

        self.net_d_left_eye = FacialComponentDiscriminator()
        self.net_d_right_eye = FacialComponentDiscriminator()
        self.net_d_mouth = FacialComponentDiscriminator()

    def forward(self, left_eyes, right_eyes, mouths, return_feats=False, **kwargs):
        fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(left_eyes, return_feats)
        fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(right_eyes, return_feats)
        fake_mouth, fake_mouth_feats = self.net_d_mouth(mouths, return_feats)


        return fake_left_eye, fake_left_eye_feats, fake_right_eye, fake_right_eye_feats,\
        fake_mouth, fake_mouth_feats


def get_optimizer_d(optim_type, params, lr, **kwargs):
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(params, lr, **kwargs)
    else:
        raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
    return optimizer

def get_roi_regions(opt, gt, output, loc_left_eyes, loc_right_eyes, loc_mouths, eye_out_size=80, mouth_out_size=120):
    face_ratio = int(opt['out_size'] / 256)
    eye_out_size *= face_ratio
    mouth_out_size *= face_ratio
    
    rois_eyes = []
    rois_mouths = []
    for b in range(loc_left_eyes.size(0)):  # loop for batch size
        # left eye and right eye
        img_inds = loc_left_eyes.new_full((2, 1), b)
        bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
        rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
        rois_eyes.append(rois)
        # mouse
        img_inds = loc_left_eyes.new_full((1, 1), b)
        rois = torch.cat([img_inds, loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
        rois_mouths.append(rois)

    rois_eyes = torch.cat(rois_eyes, 0) # .to(self.device)
    rois_mouths = torch.cat(rois_mouths, 0) # .to(self.device)

    # real images
    all_eyes = roi_align(gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes_gt = all_eyes[0::2, :, :, :]
    right_eyes_gt = all_eyes[1::2, :, :, :]
    mouths_gt = roi_align(gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
    # output
    all_eyes = roi_align(output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
    left_eyes = all_eyes[0::2, :, :, :]
    right_eyes = all_eyes[1::2, :, :, :]
    mouths = roi_align(output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

    return left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes, right_eyes, mouths

def gram_mat(x):
    """Calculate Gram matrix.

    Args:
        x (torch.Tensor): Tensor with shape of (n, c, h, w).

    Returns:
        torch.Tensor: Gram matrix.
    """
    n, c, h, w = x.size()
    features = x.view(n, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram

def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    return loss


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='config/pirender/facerender_discriminator.yaml')
    parser.add_argument('--name', default='facerender_discriminator')
    parser.add_argument('--checkpoints_dir', default='checkpoints/PIRender',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=400000)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    opt = Config(args.config, args, is_train=True)

    loc_left_eye = torch.rand((1, 4), dtype=torch.float32).cuda()
    loc_right_eye = torch.rand((1, 4), dtype=torch.float32).cuda()
    loc_mouth = torch.rand((1, 4), dtype=torch.float32).cuda()


    gt = torch.rand((1, 3, 512, 512), dtype=torch.float32).cuda()
    output = torch.rand((1, 3, 512, 512), dtype=torch.float32).cuda()


    cri_component = build_loss(opt['gan_component_opt']).to('cuda')
    cri_l1 = build_loss(opt['L1_opt']).to('cuda')
    cri_gan = build_loss(opt['gan_opt']).to('cuda')

    optim_type = opt['optim_component'].pop('type')
    lr = opt['optim_component']['lr']

    allfacialcomponentdiscriminator = AllFacialComponentDiscriminator().cuda()

    # left eye
    optimizer_d_facial = get_optimizer_d(
        optim_type,allfacialcomponentdiscriminator.parameters(), lr, betas=(0.9, 0.99))
    
    for p in allfacialcomponentdiscriminator.parameters():
        p.requires_grad = False

    l_g_total = 0
    loss_dict = OrderedDict()
    
    left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes, right_eyes, mouths \
        = get_roi_regions(opt, gt, output, loc_left_eye, loc_right_eye, loc_mouth, \
                          eye_out_size=80, mouth_out_size=120)

    fake_left_eye, fake_left_eye_feats, fake_right_eye, fake_right_eye_feats,\
        fake_mouth, fake_mouth_feats = allfacialcomponentdiscriminator(left_eyes, right_eyes, mouths, return_feats=True) # torch.Size([1, 3, 80, 80])
    l_g_gan = cri_component(fake_left_eye, True, is_disc=False)
    l_g_total += l_g_gan
    loss_dict['l_g_gan_left_eye'] = l_g_gan

    l_g_gan = cri_component(fake_right_eye, True, is_disc=False)
    l_g_total += l_g_gan
    loss_dict['l_g_gan_right_eye'] = l_g_gan

    l_g_gan = cri_component(fake_mouth, True, is_disc=False)
    l_g_total += l_g_gan
    loss_dict['l_g_gan_mouth'] = l_g_gan


    if opt['comp_style_weight'] > 0:
        # get gt feat
        real_left_eye, real_left_eye_feats, real_right_eye, real_right_eye_feats,\
        real_mouth, real_mouth_feats = allfacialcomponentdiscriminator(left_eyes_gt, right_eyes_gt, mouths_gt, return_feats=True)

        def _comp_style(feat, feat_gt, criterion):
            return criterion(gram_mat(feat[0]), gram_mat(
                feat_gt[0].detach())) * 0.5 + criterion(
                    gram_mat(feat[1]), gram_mat(feat_gt[1].detach()))

        # facial component style loss
        comp_style_loss = 0
        comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, cri_l1)
        comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, cri_l1)
        comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, cri_l1)
        comp_style_loss = comp_style_loss * opt['comp_style_weight']
        l_g_total += comp_style_loss
        loss_dict['l_g_comp_style_loss'] = comp_style_loss
    
    # optimize facial component discriminators

    # left eye


    fake_d_pred, _ = allfacialcomponentdiscriminator.net_d_left_eye(left_eyes.detach())
    real_d_pred, _ = allfacialcomponentdiscriminator.net_d_left_eye(left_eyes_gt)
    l_d_left_eye = cri_component(
        real_d_pred, True, is_disc=True) + cri_gan(
            fake_d_pred, False, is_disc=True)
    loss_dict['l_d_left_eye'] = l_d_left_eye
    l_d_left_eye.backward()
    # right eye
    fake_d_pred, _ = allfacialcomponentdiscriminator.net_d_right_eye(right_eyes.detach())
    real_d_pred, _ = allfacialcomponentdiscriminator.net_d_right_eye(right_eyes_gt)
    l_d_right_eye = allfacialcomponentdiscriminator.cri_component(
        real_d_pred, True, is_disc=True) + cri_gan(
            fake_d_pred, False, is_disc=True)
    loss_dict['l_d_right_eye'] = l_d_right_eye
    l_d_right_eye.backward()
    # mouth
    fake_d_pred, _ = allfacialcomponentdiscriminator.net_d_mouth(mouths.detach())
    real_d_pred, _ = allfacialcomponentdiscriminator.net_d_mouth(mouths_gt)
    l_d_mouth = cri_component(
        real_d_pred, True, is_disc=True) + cri_gan(
            fake_d_pred, False, is_disc=True)
    loss_dict['l_d_mouth'] = l_d_mouth
    l_d_mouth.backward()

    optimizer_d_facial.step()
