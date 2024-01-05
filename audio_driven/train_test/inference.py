import os, sys

sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
# os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
from audio_driven.models.model import *
from style_extraction.models.vqgan.vqmodules.gan_models import VQModelTransformer_encoder
from collections import OrderedDict
from audio_driven.models.audio_encoder import *

from audio2pose.models.modules.fact_model import setup_model
from audio2pose.models.utils.base_model_util import *
from audio2pose.models.utils.load_utils import *

from style_extraction.models.vqgan.vqmodules.gan_models import setup_vq_transformer


def multi2single(pretrain):
    new_state_dict = OrderedDict()
    for key, value in pretrain.items():
        name = key[7:]
        new_state_dict[name] = value
    return new_state_dict


def load_model(args, load_path):
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)

    style_encoder = VQModelTransformer_encoder(config, version=None).cuda()

    audio_encoder = AudioEncoder2(args).cuda()

    ra = ResidualAdapter(args).cuda()

    decoder = Decoder(args).cuda()


    if os.path.exists(load_path):
        checkpoints = torch.load(load_path)
        audio_encoder.load_state_dict(checkpoints['audio_encoder'])
        style_encoder.load_state_dict(checkpoints['style_encoder'])
        ra.load_state_dict(checkpoints['ra'])
        decoder.load_state_dict(checkpoints['decoder'])


    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)
    l_model_path = args.pose_codebook_checkpoint
    l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path)
    for param in l_vq_model.parameters():
        param.requires_grad = False
    l_vq_model.eval()

    load_path = args.audio2pose_checkpoint
    generator, g_optimizer, start_epoch = setup_model(config, l_vqconfig,
                                                      s_vqconfig=None,
                                                        load_path=load_path)
    generator.eval()
    return config, audio_encoder, style_encoder, ra, decoder, l_vq_model, generator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



def gen_expression(audio_encoder, style_encoder, ra, decoder, source_example_parameters, source_audio_feature, target_parameters):

    source_style_code = style_encoder(source_example_parameters[:,80:144].unsqueeze(1).repeat(1,32,1))
    target_style_code = style_encoder(target_parameters[:,:,80:144])
    z_t = audio_encoder(source_audio_feature,source_example_parameters, source_style_code)
    z_t_prime = ra(z_t, target_style_code)
    results = decoder(z_t_prime, target_style_code)

    return results

import librosa
import python_speech_features

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
def gen_pose(config, l_vq_model, style_encoder, audio2pose, target_parameters, wav_path):
    audio_feature = generate_mfcc(wav_path).type(torch.FloatTensor).cuda()

    target_len = min(target_parameters.shape[1], audio_feature.shape[1])

    target_parameters = target_parameters[:,:target_len]
    target_audio_feature = audio_feature[:,:target_len]

    cur_len = target_len
    while (int(cur_len/32)*32 != cur_len):
        target_audio_feature = torch.cat([target_audio_feature, target_audio_feature[:,(cur_len-1):cur_len]],1)
        target_parameters = torch.cat([target_parameters, target_parameters[:,(cur_len-1):cur_len]],1)
        cur_len = target_parameters.shape[1]
    
    seq_len = 32
    speakerData_np = target_parameters[:,:seq_len,80:144]
    speakerData_np = style_encoder(speakerData_np)
    listenerData_np = torch.cat([target_parameters[:,:,224:227], target_parameters[:,:,254:257]],2)
    cut_point = config['fact_model']['listener_past_transformer_config']\
                    ['sequence_length']

    inputs, _, raw_listener, quant_size = \
        create_data_vq_test(l_vq_model, speakerData_np, listenerData_np[:,:seq_len],
                        target_audio_feature[:,:seq_len], seq_len,
                        data_type=config['loss_config']['loss_type'])
    quant_prediction = audio2pose(inputs,
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
        quant_prediction = audio2pose(inputs,
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

    prediction = decoded_pred

    output_pred = savgol_filter(np.array(prediction.cpu()), 13, 2, axis=1) 
    output_pred = output_pred[:,:target_len]
    return output_pred


from PIL import Image


def write2video(results_dir, *video_list):
    cat_video=None

    for video in video_list:
        video_numpy = video[:,:3,:,:].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i])

    out_name = results_dir
    _, height, width, layers = cat_video.shape
    size = (width,height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

    for i in range(len(image_array)):
        out.write(image_array[i][:,:,::-1])
    out.release() 

def add_audio(video_name=None, audio_dir = None):

    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    print (command)
    
    os.system(command)
    os.remove(video_name)


def obtain_seq_index(index, num_frames, radius):
    seq = list(range(index - radius, index + radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

@torch.no_grad()
def render_video(
    net_G, src_img_path, target_exp_seq, wav_path, output_path, silent=False, semantic_radius=13, fps=30, split_size=64
):

    # target_exp_seq = np.load(exp_path)

    frame = cv2.imread(src_img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    src_img_raw = Image.fromarray(frame)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    src_img = image_transform(src_img_raw)

    target_win_exps = []
    for frame_idx in range(len(target_exp_seq)):
        win_indices = obtain_seq_index(frame_idx, target_exp_seq.shape[0], semantic_radius)
        win_exp = torch.tensor(target_exp_seq[win_indices]).permute(1, 0)
        # (73, 27)
        target_win_exps.append(win_exp)

    target_exp_concat = torch.stack(target_win_exps, dim=0)
    target_splited_exps = torch.split(target_exp_concat, split_size, dim=0)
    output_imgs = []
    for win_exp in target_splited_exps:
        win_exp = win_exp.cuda()
        cur_src_img = src_img.expand(win_exp.shape[0], -1, -1, -1).cuda()
        output_dict = net_G(cur_src_img, win_exp.float())
        output_imgs.append(output_dict["fake_image"].cpu().clamp_(-1, 1))

    output_imgs = torch.cat(output_imgs, 0)


    write2video(output_path, output_imgs)
    
    if not silent:
        add_audio(output_path, wav_path)

def generator_full(config, encoder, style_encoder, ra, decoder, l_vq_model, audio2pose,source_example_parameters, \
                       source_audio_feature, target_parameters, wav_path):

    encoder.eval()
    style_encoder.eval()
    ra.eval()
    decoder.eval()


    source_example_parameters = source_example_parameters.type(torch.FloatTensor).cuda()
    source_audio_feature = source_audio_feature.type(torch.FloatTensor).cuda()

    target_parameters = target_parameters.type(torch.FloatTensor).cuda()

    style_len = target_parameters.shape[1]
    start_r = random.choice([x for x in range(style_len-32)])
    a_b_expression = gen_expression(encoder, style_encoder, ra, decoder, source_example_parameters, source_audio_feature, target_parameters[:,start_r:start_r+32])
    
    a_b_expression = np.array(a_b_expression[0].detach().cpu().numpy())

    a_b_pose = gen_pose(config, l_vq_model, style_encoder, audio2pose, target_parameters, wav_path)[0]

    if len(a_b_pose)<len(a_b_expression):

        gap = len(a_b_expression)-len(a_b_pose)
        n = int((gap/len(a_b_pose)/2)) +2
        a_b_pose = np.concatenate((a_b_pose,a_b_pose[::-1,:]),axis = 0)
        a_b_pose = np.tile(a_b_pose, (n,1))

    a_b_pose = a_b_pose[:len(a_b_expression)]

    a_b = np.concatenate((a_b_expression, a_b_pose), axis=-1)

    return a_b




def generate_beta(config, encoder, style_encoder, ra, decoder, l_vq_model, audio2pose, img_path, wav_path, style_path):
    source_coeffs_pred_numpy = np.load(img_path, allow_pickle=True)
    source_coeffs_pred_numpy = dict(enumerate(source_coeffs_pred_numpy.flatten(), 0))[0]
    source_coeff = source_coeffs_pred_numpy['coeff']
    source_coeff_mouth = source_coeff

    source_example_parameters = torch.from_numpy(np.array(source_coeff_mouth[0])).unsqueeze(0).cuda()
    source_audio_feature, source_nums = get_mel(wav_path)


    target_coeffs_pred_numpy = np.load(style_path, allow_pickle=True)
    target_coeffs_pred_numpy = dict(enumerate(target_coeffs_pred_numpy.flatten(), 0))[0]
    target_coeff = target_coeffs_pred_numpy['coeff']
    target_coeff_mouth = target_coeff

    target_parameters = torch.from_numpy(np.array(target_coeff_mouth)).unsqueeze(0).cuda()


    a_b = \
        generator_full(config, encoder, style_encoder, ra, decoder, l_vq_model, audio2pose,source_example_parameters, \
                       source_audio_feature, target_parameters, wav_path)

    transform_params = source_coeffs_pred_numpy['trans_param']
    _, _, ratio, t0, t1 = np.hsplit(transform_params.astype(np.float32), 5)

    if len(ratio)<len(a_b):

        gap = len(a_b)-len(ratio)
        n = int((gap/len(ratio)/2)) +2
        ratio = np.concatenate((ratio,ratio[::-1,:]),axis = 0)
        ratio = np.tile(ratio, (n,1))

        t0 = np.concatenate((t0,t0[::-1,:]),axis = 0)
        t0 = np.tile(t0, (n,1))

        t1 = np.concatenate((t1,t1[::-1,:]),axis = 0)
        t1 = np.tile(t1, (n,1))
    
    ratio = ratio[:len(a_b)]
    t0 = t0[:len(a_b)]
    t1 = t1[:len(a_b)]

    a_b = np.concatenate([a_b, ratio, t0, t1], 1)
    return a_b

@torch.no_grad()
def get_netG(checkpoint_path):
    from audio_driven.models.face_model import FaceGenerator
    import yaml

    with open("configs/pirender/renderer_conf.yaml", "r") as f:
        renderer_config = yaml.load(f, Loader=yaml.FullLoader)

    renderer = FaceGenerator(**renderer_config).to(torch.cuda.current_device())

    if os.path.exists(checkpoint_path)==False:
        renderer.eval()
        return renderer

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    renderer.load_state_dict(checkpoint["net_G_ema"], strict=False)

    renderer.eval()

    return renderer

def main(args):

    load_path = args.audio_driven_checkpoint
    config, encoder, style_encoder, ra, decoder, l_vq_model, audio2pose = load_model(args, load_path)
    encoder.eval()
    style_encoder.eval()
    ra.eval()
    decoder.eval()

    image_renderer = get_netG(args.pirender_checkpoint)
    print('model loaded!')


    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir,exist_ok=True)

    with torch.no_grad():
        beta_hat = generate_beta(config, encoder, style_encoder, ra, decoder, l_vq_model, audio2pose, args.img_3DMM_path, args.wav_path, args.style_path)


    render_video(
        image_renderer,
        args.img_path,
        beta_hat,
        args.wav_path,
        args.save_path,
        split_size=4,
    )


class ArgParserTest(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.add_argument('--local_rank', type=int, default=0)
        self.add_argument("--encoder_layer_num", type=int, default=2)
        self.add_argument("--decoder_layer_num", type=int, default=4)
        self.add_argument("--discriminator_layer_num", type=int, default=4)
        self.add_argument("--classifier_layer_num", type=int, default=5)
        self.add_argument("--latent_dim", type=int, default=64)
        self.add_argument("--style_dim", type=int, default=256)
        self.add_argument("--expression_dim", type=int, default=64)
        self.add_argument("--cov_dim", type=int, default=128)
        self.add_argument("--ra_input_dim", type=int, default=320)
        self.add_argument("--neutral_layer_num", type=int, default=4)
        self.add_argument("--style_layer_num", type=int, default=6)
        self.add_argument("--feature_dim", type=int, default=16)

        self.add_argument('--config', type=str, default='configs/audio_driven/audio_driven.json')
        self.add_argument('--audio_driven_checkpoint', type=str, default='checkpoints/audio_driven/audio_driven.pth')
        self.add_argument('--audio2pose_checkpoint', type=str, default='checkpoints/audio2pose/audio2pose.pth')
        self.add_argument('--pose_codebook_checkpoint', type=str, default='checkpoints/audio2pose/pose_codebook.pth')
        self.add_argument('--pirender_checkpoint', type=str, default='checkpoints/pirender/checkpoint.pt')

        self.add_argument('--img_path', type=str, default='demo/audio_driven/source/image/test.jpg')
        self.add_argument('--wav_path', type=str, default='demo/audio_driven/source/audio/test.wav')
        self.add_argument('--img_3DMM_path', type=str, default='demo/audio_driven/source/image_3DMM/test.npy')
        self.add_argument('--style_path', type=str, default='demo/audio_driven/source/style_clip_3DMM/test.npy')
        self.add_argument('--save_path', type=str, default='demo/audio_driven/results/test.mp4')



if __name__ == '__main__':
    args = ArgParserTest().parse_args()
    main(args)
