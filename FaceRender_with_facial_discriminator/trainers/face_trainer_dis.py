import math

import torch

from trainers.base_dis import BaseTrainer
from util.trainer import accumulate, get_optimizer
from loss.perceptual  import PerceptualLoss
from generators.FacialComponentDiscriminator import *


class FaceTrainer(BaseTrainer):
    r"""Initialize lambda model trainer.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        opt_G (obj): Optimizer for the generator network.
        sch_G (obj): Scheduler for the generator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self, opt, net_G, 
                 net_d_left_eye, net_d_right_eye, net_d_mouth,
                 net_G_ema,
                 opt_G,
                 sch_G, 
                 opt_d_left_eye, opt_d_right_eye, opt_d_mouth,
                 train_data_loader, val_data_loader=None):
        super(FaceTrainer, self).__init__(opt, net_G, 
                 net_d_left_eye, net_d_right_eye, net_d_mouth,
                 net_G_ema,
                 opt_G,
                 sch_G, 
                 opt_d_left_eye, opt_d_right_eye, opt_d_mouth,
                 train_data_loader, val_data_loader)
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        self.cri_component = build_loss(opt.gan_component_opt).to('cuda')
        self.cri_l1 = build_loss(opt.L1_opt).to('cuda')
        self.cri_gan = build_loss(opt.gan_opt).to('cuda')
        self.distributed = opt.distributed


    def _init_loss(self, opt):
        self._assign_criteria(
            'perceptual_warp',
            PerceptualLoss(
                network=opt.trainer.vgg_param_warp.network,
                layers=opt.trainer.vgg_param_warp.layers,
                num_scales=getattr(opt.trainer.vgg_param_warp, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_warp, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_warp, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_warp)

        self._assign_criteria(
            'perceptual_final',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_final)



    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data):
        self.gen_losses = {}
        source_image, target_image = data['source_image'], data['target_image']
        source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

        input_image = torch.cat((source_image, target_image), 0)
        input_semantic = torch.cat((target_semantic, source_semantic), 0)
        gt_image = torch.cat((target_image, source_image), 0) 

        output_dict = self.net_G(input_image, input_semantic, self.training_stage)

        if self.training_stage == 'gen':
            fake_img = output_dict['fake_image']
            warp_img = output_dict['warp_image']
            self.gen_losses["perceptual_final"] = self.criteria['perceptual_final'](fake_img, gt_image)
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](warp_img, gt_image)
        else:
            warp_img = output_dict['warp_image']
            self.gen_losses["perceptual_warp"] = self.criteria['perceptual_warp'](warp_img, gt_image)

        for p in self.net_d_left_eye.parameters():
            p.requires_grad = False
        for p in self.net_d_right_eye.parameters():
            p.requires_grad = False
        for p in self.net_d_mouth.parameters():
            p.requires_grad = False

        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss


        input_left_eye = torch.cat((data['source_loc_left_eye'], data['target_loc_left_eye']), 0)
        input_right_eye = torch.cat((data['source_loc_right_eye'], data['target_loc_right_eye']), 0)
        input_mouth = torch.cat((data['source_loc_mouth'], data['target_loc_mouth']), 0)
        
        # 根据daataets 要concat一下
        left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes, right_eyes, mouths \
            = get_roi_regions(self.opt, gt_image, output_dict['fake_image'], input_left_eye, input_right_eye, input_mouth, \
                            eye_out_size=40, mouth_out_size=60)


        fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(left_eyes, return_feats=True)
        fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(right_eyes, return_feats=True)
        fake_mouth, fake_mouth_feats = self.net_d_mouth(mouths, return_feats=True)


        l_g_gan = self.cri_component(fake_left_eye, True, is_disc=False)
        total_loss += l_g_gan
        self.gen_losses['l_g_gan_left_eye'] = l_g_gan

        l_g_gan = self.cri_component(fake_right_eye, True, is_disc=False)
        total_loss += l_g_gan
        self.gen_losses['l_g_gan_right_eye'] = l_g_gan

        l_g_gan = self.cri_component(fake_mouth, True, is_disc=False)
        total_loss += l_g_gan
        self.gen_losses['l_g_gan_mouth'] = l_g_gan


        if self.opt.comp_style_weight > 0:
            # get gt feat

            real_left_eye, real_left_eye_feats = self.net_d_left_eye(left_eyes, return_feats=True)
            real_right_eye, real_right_eye_feats = self.net_d_right_eye(right_eyes, return_feats=True)
            real_mouth, real_mouth_feats = self.net_d_mouth(mouths, return_feats=True)

            def _comp_style(feat, feat_gt, criterion):
                return criterion(gram_mat(feat[0]), gram_mat(
                    feat_gt[0].detach())) * 0.5 + criterion(
                        gram_mat(feat[1]), gram_mat(feat_gt[1].detach()))

            # facial component style loss
            comp_style_loss = 0
            comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.cri_l1)
            comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.cri_l1)
            comp_style_loss = comp_style_loss * self.opt.comp_style_weight
            total_loss += comp_style_loss
            self.gen_losses['l_g_comp_style_loss'] = comp_style_loss
        

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

        accumulate(self.net_G_ema, self.net_G_module, self.accum)


    def optimize_parameters_d(self, data):
        self.opt_d_left_eye.zero_grad()
        self.opt_d_right_eye.zero_grad()
        self.opt_d_mouth.zero_grad()

        for p in self.net_d_left_eye.parameters():
            p.requires_grad = True
        for p in self.net_d_right_eye.parameters():
            p.requires_grad = True
        for p in self.net_d_mouth.parameters():
            p.requires_grad = True

        self.dis_losses = {}
        source_image, target_image = data['source_image'], data['target_image']
        source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

        input_image = torch.cat((source_image, target_image), 0)
        input_semantic = torch.cat((target_semantic, source_semantic), 0)
        gt_image = torch.cat((target_image, source_image), 0) 

        output_dict = self.net_G(input_image, input_semantic, self.training_stage)

    
        input_left_eye = torch.cat((data['source_loc_left_eye'], data['target_loc_left_eye']), 0)
        input_right_eye = torch.cat((data['source_loc_right_eye'], data['target_loc_right_eye']), 0)
        input_mouth = torch.cat((data['source_loc_mouth'], data['target_loc_mouth']), 0)
        
        # 根据daataets 要concat一下
        left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes, right_eyes, mouths \
            = get_roi_regions(self.opt, gt_image, output_dict['fake_image'], input_left_eye, input_right_eye, input_mouth, \
                            eye_out_size=40, mouth_out_size=60)


        fake_d_pred, _ = self.net_d_left_eye(left_eyes.detach())
        real_d_pred, _ = self.net_d_left_eye(left_eyes_gt)
        l_d_left_eye = self.cri_component(
            real_d_pred, True, is_disc=True) + self.cri_gan(
                fake_d_pred, False, is_disc=True)
        self.dis_losses['l_d_left_eye'] = l_d_left_eye

        l_d_left_eye.backward()
        # right eye
        fake_d_pred, _ = self.net_d_right_eye(right_eyes.detach())
        real_d_pred, _ = self.net_d_right_eye(right_eyes_gt)
        l_d_right_eye = self.cri_component(
            real_d_pred, True, is_disc=True) + self.cri_gan(
                fake_d_pred, False, is_disc=True)
        self.dis_losses['l_d_right_eye'] = l_d_right_eye

        l_d_right_eye.backward()
        # mouth
        fake_d_pred, _ = self.net_d_mouth(mouths.detach())
        real_d_pred, _ = self.net_d_mouth(mouths_gt)
        l_d_mouth = self.cri_component(
            real_d_pred, True, is_disc=True) + self.cri_gan(
                fake_d_pred, False, is_disc=True)
        self.dis_losses['l_d_mouth'] = l_d_mouth

        l_d_mouth.backward()

        
        self.opt_d_left_eye.step()
        self.opt_d_right_eye.step()
        self.opt_d_mouth.step()

    def _start_of_iteration(self, data, current_iteration):
        self.training_stage = 'gen' if current_iteration >= self.opt.trainer.pretrain_warp_iteration else 'warp'
        if current_iteration == self.opt.trainer.pretrain_warp_iteration:
            self.reset_trainer()
        return data

    def reset_trainer(self):
        if self.distributed:
            self.opt_G = get_optimizer(self.opt.gen_optimizer, self.net_G.module)
        else:
            self.opt_G = get_optimizer(self.opt.gen_optimizer, self.net_G)

    def _get_visualizations(self, data):
        source_image, target_image = data['source_image'], data['target_image']
        source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

        input_image = torch.cat((source_image, target_image), 0)
        input_semantic = torch.cat((target_semantic, source_semantic), 0)        
        with torch.no_grad():
            self.net_G_ema.eval()
            output_dict = self.net_G_ema(
                input_image, input_semantic, self.training_stage
                )
            if self.training_stage == 'gen':
                fake_img = torch.cat([output_dict['warp_image'], output_dict['fake_image']], 3)
            else:
                fake_img = output_dict['warp_image']

            fake_source, fake_target = torch.chunk(fake_img, 2, dim=0)
            sample_source = torch.cat([source_image, fake_source, target_image], 3)
            sample_target = torch.cat([target_image, fake_target, source_image], 3)                    
            sample = torch.cat([sample_source, sample_target], 2)
            sample = torch.cat(torch.chunk(sample, sample.size(0), 0)[:3], 2)
        return sample

    def test(self, data_loader, output_dir, current_iteration=-1):
        pass

    def _compute_metrics(self, data, current_iteration):
        if self.training_stage == 'gen':
            source_image, target_image = data['source_image'], data['target_image']
            source_semantic, target_semantic = data['source_semantics'], data['target_semantics']

            input_image = torch.cat((source_image, target_image), 0)
            input_semantic = torch.cat((target_semantic, source_semantic), 0)        
            gt_image = torch.cat((target_image, source_image), 0)        
            metrics = {}
            with torch.no_grad():
                self.net_G_ema.eval()
                output_dict = self.net_G_ema(
                    input_image, input_semantic, self.training_stage
                    )
                fake_image = output_dict['fake_image']
                metrics['lpips'] = self.lpips(fake_image, gt_image).mean()
            return metrics