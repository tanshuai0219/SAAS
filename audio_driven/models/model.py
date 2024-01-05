import torch, sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
import torch.nn as nn
import time
from torch import autograd
import numpy as np
import torch, json
from scipy.signal import hann
import torch.nn.functional as F
import os
from style_extraction.models.vqgan.vqmodules.gan_models import setup_style_encoder, setup_style_discriminator
from style_extraction.models.utils.load_utils import *
from audio_driven.models.audio_encoder import *


import argparse


def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class HyperNetwork(nn.Module):
    def __init__(self):
        super(HyperNetwork, self).__init__()
        
        self.inc = (DoubleConv(1, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        return x3



class AudioEncoder2(nn.Module):
    def __init__(self, args):
        super().__init__()

        netG = SimpleWrapperV2()
        netG.cuda()
        self.audioencoder = AudioEncoder(netG, device='cuda', prepare_training_loss=False)

        layers = []
        input_size = 64+args.style_dim
        for _ in range(args.encoder_layer_num):
            layers.append(nn.Linear(input_size, input_size // 4 * 2))
            layers.append(nn.ReLU())
            input_size = input_size // 4 * 2

        layers.append(nn.Linear(input_size, args.latent_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, source_audio_feature,source_example_parameters, style_code):
        ratio = generate_blink_seq_randomly(source_audio_feature.shape[1])
        ratio = torch.FloatTensor(ratio).unsqueeze(0).cuda()        
        
        audio_feature = self.audioencoder(source_audio_feature, source_example_parameters[:,80:144].unsqueeze(1), ratio)

        style_code = style_code.unsqueeze(1).repeat(1,audio_feature.shape[1],1)

        return self.layers(torch.cat((audio_feature, style_code), dim=-1))



class ResidualAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.neutral_branch = nn.LSTM(input_size=args.latent_dim, hidden_size=args.latent_dim,
                                      num_layers=args.neutral_layer_num, batch_first=True)
        self.neutral_init_hidden_state = nn.Parameter(
            torch.zeros(args.neutral_layer_num, args.latent_dim))
        self.neutral_init_cell_state = nn.Parameter(
            torch.zeros(args.neutral_layer_num, args.latent_dim))


        self.ra_style = nn.LSTM(input_size=args.ra_input_dim, hidden_size=args.latent_dim,
                        num_layers=args.style_layer_num, batch_first=True)
        
        self.ra_style_tmp = nn.LSTM(input_size=args.ra_input_dim, hidden_size=args.latent_dim,
                                num_layers=args.style_layer_num, batch_first=True)
        self.ra_init_hidden_state = nn.Linear(args.style_dim, args.style_layer_num*args.latent_dim)
        self.ra_init_cell_state = nn.Linear(args.style_dim, args.style_layer_num*args.latent_dim)

        self.args = args

        self.linear_layers_ih = nn.ModuleList(
            [HyperNetwork() for _ in range(4)]
        )

        self.linear_layers_hh = nn.ModuleList(
            [HyperNetwork() for _ in range(4)]
        )

        self.linear_layers_bias_ih = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(4)]
        )
        self.linear_layers_bias_hh = nn.ModuleList(
            [nn.Linear(256, 256) for _ in range(4)]
        )

    def forward(self, latent_code, transferred_style, source_style = None):
        batch_size, T, _ = latent_code.shape # torch.Size([1, 80, 32])

        transferred_style_T = transferred_style.unsqueeze(dim = 1).repeat(1,latent_code.shape[1],1)
        


        h0_neutral = self.neutral_init_hidden_state.unsqueeze(1).repeat(1, batch_size, 1)
        c0_neutral = self.neutral_init_cell_state.unsqueeze(1).repeat(1, batch_size, 1)
        neutral_output, (hn, cn) = self.neutral_branch(latent_code, (h0_neutral, c0_neutral)) # torch.Size([1, 80, 32])

        params_ih = []
        params_bias_ih = []
        params_hh = []
        params_bias_hh = []
        for linear_layer in self.linear_layers_ih:
            params_ih.append(linear_layer(transferred_style))

        for linear_layer in self.linear_layers_hh:
            params_hh.append(linear_layer(transferred_style))

        for linear_layer in self.linear_layers_bias_ih:
            params_bias_ih.append(linear_layer(transferred_style))

        for linear_layer in self.linear_layers_bias_hh:
            params_bias_hh.append(linear_layer(transferred_style))
 
        # if test_time:
        h0_ra = self.ra_init_hidden_state(transferred_style).reshape(batch_size, self.args.style_layer_num,self.args.latent_dim).permute(1, 0, 2).contiguous()
        c0_ra = self.ra_init_cell_state(transferred_style).reshape(batch_size, self.args.style_layer_num,self.args.latent_dim).permute(1, 0, 2).contiguous()
        
        batch_size = transferred_style.shape[0]

        final_result = []#self.ra_style(torch.cat((latent_code, transferred_style_T), 2), (h0_ra, c0_ra))[0]
       
        for i in range(batch_size):

            self.ra_style_tmp.weight_ih_l1.data = self.ra_style.weight_ih_l1.data * (1+params_ih[0][i])
            # print(self.ra_style.weight_ih_l1.data.shape)
            self.ra_style_tmp.weight_hh_l1.data = self.ra_style.weight_hh_l1.data* (1+params_hh[0][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_ih_l1.data = self.ra_style.bias_ih_l1.data* (1+params_bias_ih[0][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_hh_l1.data = self.ra_style.bias_hh_l1.data* (1+params_bias_hh[0][i])

            self.ra_style_tmp.weight_ih_l2.data = self.ra_style.weight_ih_l2.data* (1+params_ih[1][i])
            # print(self.ra_style.weight_ih_l1.data.shape)
            self.ra_style_tmp.weight_hh_l2.data = self.ra_style.weight_hh_l2.data* (1+params_hh[1][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_ih_l2.data = self.ra_style.bias_ih_l2.data* (1+params_bias_ih[1][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_hh_l2.data = self.ra_style.bias_hh_l2.data * (1+params_bias_hh[1][i])

            
            self.ra_style_tmp.weight_ih_l3.data = self.ra_style.weight_ih_l3.data* (1+params_ih[2][i])
            # print(self.ra_style.weight_ih_l1.data.shape)
            self.ra_style_tmp.weight_hh_l3.data = self.ra_style.weight_hh_l3.data* (1+params_hh[2][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_ih_l3.data = self.ra_style.bias_ih_l3.data* (1+params_bias_ih[2][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_hh_l3.data = self.ra_style.bias_hh_l3.data* (1+params_bias_hh[2][i])

            self.ra_style_tmp.weight_ih_l4.data =self.ra_style.weight_ih_l4.data * (1+params_ih[3][i])
            # print(self.ra_style.weight_ih_l1.data.shape)
            self.ra_style_tmp.weight_hh_l4.data =self.ra_style.weight_hh_l4.data * (1+params_hh[3][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_ih_l4.data = self.ra_style.bias_ih_l4.data* (1+params_bias_ih[3][i])
            # print(self.ra_style.bias_ih_l1.data.shape)
            self.ra_style_tmp.bias_hh_l4.data = self.ra_style.bias_hh_l4.data* (1+params_bias_hh[3][i])


# 
            ra_result = self.ra_style_tmp(torch.cat((latent_code[i:i+1], transferred_style_T[i:i+1]), 2)
                                      , (h0_ra[:,i:i+1,:].contiguous(), c0_ra[:,i:i+1,:].contiguous()))[0]
            final_result.append(ra_result)

        final_result = torch.stack(final_result, 0).squeeze(1)
        ra_result_ss = []
        if source_style!= None:
            source_style_T = source_style.unsqueeze(dim = 1).repeat(1,latent_code.shape[1],1)
            params_ih_s = []
            params_bias_ih_s = []
            params_hh_s = []
            params_bias_hh_s = []
            for linear_layer in self.linear_layers_ih:
                params_ih_s.append(linear_layer(source_style))

            for linear_layer in self.linear_layers_hh:
                params_hh_s.append(linear_layer(source_style))

            for linear_layer in self.linear_layers_bias_ih:
                params_bias_ih_s.append(linear_layer(source_style))

            for linear_layer in self.linear_layers_bias_hh:
                params_bias_hh_s.append(linear_layer(source_style))
            h0_ra_s = self.ra_init_hidden_state(source_style).reshape(batch_size, self.args.style_layer_num,self.args.latent_dim).permute(1, 0, 2).contiguous()
            c0_ra_s = self.ra_init_cell_state(source_style).reshape(batch_size, self.args.style_layer_num,self.args.latent_dim).permute(1, 0, 2).contiguous()
            final_result_s = []
            for i in range(batch_size):

                self.ra_style_tmp.weight_ih_l1.data = self.ra_style.weight_ih_l1.data * (1+params_ih_s[0][i])
                # print(self.ra_style.weight_ih_l1.data.shape)
                self.ra_style_tmp.weight_hh_l1.data = self.ra_style.weight_hh_l1.data* (1+params_hh_s[0][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_ih_l1.data = self.ra_style.bias_ih_l1.data* (1+params_bias_ih_s[0][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_hh_l1.data = self.ra_style.bias_hh_l1.data* (1+params_bias_hh_s[0][i])

                self.ra_style_tmp.weight_ih_l2.data = self.ra_style.weight_ih_l2.data* (1+params_ih_s[1][i])
                # print(self.ra_style.weight_ih_l1.data.shape)
                self.ra_style_tmp.weight_hh_l2.data = self.ra_style.weight_hh_l2.data* (1+params_hh_s[1][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_ih_l2.data = self.ra_style.bias_ih_l2.data* (1+params_bias_ih_s[1][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_hh_l2.data = self.ra_style.bias_hh_l2.data * (1+params_bias_hh_s[1][i])

                
                self.ra_style_tmp.weight_ih_l3.data = self.ra_style.weight_ih_l3.data* (1+params_ih_s[2][i])
                # print(self.ra_style.weight_ih_l1.data.shape)
                self.ra_style_tmp.weight_hh_l3.data = self.ra_style.weight_hh_l3.data* (1+params_hh_s[2][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_ih_l3.data = self.ra_style.bias_ih_l3.data* (1+params_bias_ih_s[2][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_hh_l3.data = self.ra_style.bias_hh_l3.data* (1+params_bias_hh_s[2][i])

                self.ra_style_tmp.weight_ih_l4.data =self.ra_style.weight_ih_l4.data * (1+params_ih_s[3][i])
                # print(self.ra_style.weight_ih_l1.data.shape)
                self.ra_style_tmp.weight_hh_l4.data =self.ra_style.weight_hh_l4.data * (1+params_hh_s[3][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_ih_l4.data = self.ra_style.bias_ih_l4.data* (1+params_bias_ih_s[3][i])
                # print(self.ra_style.bias_ih_l1.data.shape)
                self.ra_style_tmp.bias_hh_l4.data = self.ra_style.bias_hh_l4.data* (1+params_bias_hh_s[3][i])




                ra_result = self.ra_style_tmp(torch.cat((latent_code[i:i+1], source_style_T[i:i+1]), 2), (h0_ra_s[:,i:i+1,:].contiguous(), c0_ra_s[:,i:i+1,:].contiguous()))[0]
                final_result_s.append(ra_result)
            ra_result_ss = torch.stack(final_result_s, 0).squeeze(1)
            return neutral_output + final_result, neutral_output + ra_result_ss


        return neutral_output + final_result



class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        current_dim = args.latent_dim + args.style_dim
        self.args = args
        layers = []
        for _ in range(args.decoder_layer_num - 1):
            layers.append(nn.Linear(current_dim, int((current_dim * 1.5 // 2) * 2)))
            layers.append(nn.ReLU())
            current_dim = int((current_dim * 1.5 // 2) * 2)

        self.features = nn.Sequential(*layers)
        self.last_layer_expression = nn.Linear(current_dim, 64-18)
        self.last_layer_mouth = nn.Linear(current_dim, 18)

    def forward(self, latent_code, target_style, test_time=False, source_style = None):
        target_style = target_style.unsqueeze(dim = 1).repeat(1,latent_code.shape[1],1)
        features = self.features(torch.cat((latent_code,target_style), dim=-1)) # torch.Size([1, 80, 128])

        output_expression = self.last_layer_expression(features)
        output_mouth = self.last_layer_mouth(features)

        output = torch.cat((output_mouth, output_expression), dim=2)

        if source_style:
            source_style = source_style.unsqueeze(dim = 1).repeat(1,latent_code.shape[1],1)
            features_s = self.features(torch.cat((latent_code,source_style), dim=-1)) # torch.Size([1, 80, 128])

            output_s_expression = self.last_layer_expression(features_s)
            output_s_mouth = self.last_layer_mouth(features_s)

            output_s = torch.cat((output_s_mouth, output_s_expression), dim=2)
            return output, output_s

        return output



class Style_Discriminator2(nn.Module):
    def __init__(self, args):
        super().__init__()
        layers = []
        current_size = 64
        dummy_data = torch.zeros(1, current_size, args['episode_length'])
        for _ in range(args['discriminator_layer_num']):
            layers.append(conv_layer(3, current_size, (current_size // 3) * 2))
            layers.append(nn.LeakyReLU())
            current_size = (current_size // 3) * 2
        self.features = nn.Sequential(*layers)

        self.last_layer = conv_layer(3, current_size, args['feature_dim'])

        input_size = args['style_class']

        self.attention_features = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )
        self.temporal_attention = nn.Linear(64, self.last_layer(self.features(dummy_data)).shape[-1])
        self.feature_attention = nn.Linear(64, args['feature_dim'])

    def forward(self, input_data, style_label, compute_grad):
        input_data = input_data.permute(0, 2, 1)
        if compute_grad: input_data.requires_grad_()
        features = self.last_layer(self.features(input_data))

        attention_input = style_label
        attention_features = self.attention_features(attention_input)
        temporal_attention = self.temporal_attention(attention_features)
        feature_attention = self.feature_attention(attention_features)

        combined_features = (features * feature_attention.unsqueeze(-1)).sum(dim=1)
        final_score = (temporal_attention * combined_features).sum(dim=-1)
        grad = None
        if compute_grad:
            batch_size = final_score.shape[0]
            grad = autograd.grad(outputs=final_score.mean(),
                                 inputs=input_data,
                                 create_graph=True,
                                 retain_graph=True,
                                 only_inputs=True)[0]
            grad = (grad ** 2).sum() / batch_size
        return final_score, grad



def apply_hanning_window(x, window_size = 3):
    window = torch.from_numpy(hann(window_size))
    window = window.to(x.device)


    pad = (window_size - 1) // 2
    padded_x = torch.cat([x[:1, :].repeat(pad, 1), x, x[-1:, :].repeat(pad, 1)], dim=0)


    output = []
    for i in range(pad, padded_x.shape[0] - pad):
        windowed = padded_x[i-pad:i+pad+1, :] * window[:, None]
        avg = torch.mean(windowed, dim=0, keepdim=True)
        output.append(avg)
    output = torch.cat(output, dim=0)


    return output

class StyleDiscriminator(nn.Module):
    def __init__(self, input_nc=64, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm1d, use_sigmoid=False, getIntermFeat=False, num_styles=256):
        super(StyleDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.num_styles = num_styles

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv1d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv1d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv1d(nf, num_styles, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        # input shape: (batch_size, sequence_len, input_nc)
        input = input.transpose(1, 2)  # transpose to (batch_size, input_nc, sequence_len)
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            output = self.model(input)  # output shape: (batch_size, num_styles, sequence_len)
            output = torch.mean(output, dim=2)  # output shape: (batch_size, num_styles)
            return output







class TemporalDiscriminator2(nn.Module):
    def __init__(self, input_size=64, L=32, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm1d, use_sigmoid=False):
        super(TemporalDiscriminator2, self).__init__()
        self.L = L
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = nn.ModuleList([nn.Conv1d(input_size, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)])

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += nn.ModuleList([
                nn.Conv1d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ])

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += nn.ModuleList([
            nn.Conv1d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ])

        sequence += nn.ModuleList([nn.Conv1d(nf, 1, kernel_size=kw, stride=1, padding=padw)])

        if use_sigmoid:
            sequence += nn.ModuleList([nn.Sigmoid()])

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # input shape: (batch_size, L, input_size)
        # output shape: (batch_size, 1)
        batch_size = input.shape[0]
        input = input.transpose(1,2) # (batch_size, input_size, L)
        input_reshaped = input.view(-1, input.size(1), self.L) # (batch_size*L, input_size, 1)
        output = self.model(input_reshaped)
        output_reshaped = output.view(batch_size, -1, 1) # (batch_size, L, 1)
        # output_mean = torch.mean(output_reshaped, dim=1) # (batch_size, 1)
        return output_reshaped
    

class TemporalDiscriminator(nn.Module):
    def __init__(self, num_frames=10):
        super(TemporalDiscriminator, self).__init__()
        
        self.num_frames = num_frames
        self.conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(512, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [batch_size, num_frames, 64]
        x = x.transpose(1, 2)  # [batch_size, 64, num_frames]
        h1 = self.conv1(x)
        h2 = self.conv2(torch.relu(h1))
        h3 = self.conv3(torch.relu(h2))
        h4 = self.conv4(torch.relu(h3))
        h5 = self.conv5(torch.relu(h4))
        out = h5.mean(dim=2)  # [batch_size, 1]
        return out

