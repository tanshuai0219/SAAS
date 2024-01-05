import os, sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
import random
from video_driven.model import *
import glob
from collections import OrderedDict
import argparse
from style_extraction.models.vqgan.vqmodules.gan_models import VQModelTransformer_encoder


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

    style_encoder = style_encoder.cuda()
    encoder = Encoder(args).cuda()
    ra = ResidualAdapter(args).cuda()
    decoder = Decoder(args).cuda()


    if os.path.exists(load_path):
        checkpoints = torch.load(load_path)

        encoder.load_state_dict(checkpoints['encoder'])
        style_encoder.load_state_dict(checkpoints['style_encoder'])
        ra.load_state_dict(checkpoints['ra'])
        decoder.load_state_dict(checkpoints['decoder'])

    return encoder, style_encoder, ra, decoder


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



def gen_test(encoder, style_encoder, ra, decoder, parameters, style_parameters):


    source_style_code = style_encoder(parameters[:,:32])
    target_style_code = style_encoder(style_parameters)
    z_t = encoder(parameters, source_style_code)
    z_t_prime = ra(z_t, target_style_code)
    results = decoder(z_t_prime, target_style_code)

    return results


def generator_test(encoder, style_encoder, ra, decoder,parameters,style_parameters):

    encoder.eval()
    style_encoder.eval()
    ra.eval()
    decoder.eval()


    parameters = parameters.type(torch.FloatTensor).cuda()

    style_parameters = style_parameters.type(torch.FloatTensor).cuda()

    a_b = gen_test(encoder, style_encoder, ra, decoder, parameters, style_parameters)

    return a_b

def generate_beta(encoder, style_encoder, ra, decoder, mouth_npy, style_npy):
    source_path = mouth_npy

    source_coeffs_pred_numpy = np.load(source_path, allow_pickle=True)
    source_coeffs_pred_numpy = dict(enumerate(source_coeffs_pred_numpy.flatten(), 0))[0]
    source_coeff = source_coeffs_pred_numpy['coeff']
    source_coeff_mouth = source_coeff

    while(len(source_coeff_mouth)<32):
        source_coeff_mouth = np.concatenate((source_coeff_mouth,source_coeff_mouth[::-1,:]),axis = 0)
 
    source_parameters = torch.from_numpy(np.array(source_coeff_mouth)).unsqueeze(0).cuda()

    target_coeffs_pred_numpy = np.load(style_npy, allow_pickle=True)
    target_coeffs_pred_numpy = dict(enumerate(target_coeffs_pred_numpy.flatten(), 0))[0]
    target_coeff = target_coeffs_pred_numpy['coeff']
    target_coeff_mouth = target_coeff

    target_len = len(target_coeff_mouth)
    while(len(target_coeff_mouth)<32):
        target_coeff_mouth = np.concatenate((target_coeff_mouth,target_coeff_mouth[::-1,:]),axis = 0)

    target_parameters = torch.from_numpy(np.array(target_coeff_mouth)).unsqueeze(0).cuda()

    style_len = target_parameters.shape[1]
    start_r = random.choice([x for x in range(style_len-32)])

    b_b = \
        generator_test(encoder, style_encoder, ra, decoder,source_parameters[:,:,80:144],target_parameters[:,start_r:start_r+32,80:144])

    b_b = b_b[0][:target_len].detach().cpu().numpy()


    transform_params = target_coeffs_pred_numpy['trans_param']
    _, _, ratio, t0, t1 = np.hsplit(transform_params.astype(np.float32), 5)

    b = np.concatenate((b_b, target_coeff[:,224:227], target_coeff[:,254:257], ratio, t0, t1),-1)

    return b



    
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

from PIL import Image
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

def main(args):
    
    load_path = args.video_driven_checkpoint

    encoder, style_encoder, ra, decoder = load_model(args, load_path)
    encoder.eval()
    style_encoder.eval()
    ra.eval()
    decoder.eval()

    image_renderer = get_netG(args.pirender_checkpoint)
    print('model loaded!')
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir,exist_ok=True)

    with torch.no_grad():
        beta_hat = generate_beta(encoder, style_encoder, ra, decoder, args.video_3DMM_path, args.style_path)


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

        self.add_argument('--config', type=str, default='configs/video_driven/video_driven.json')
        self.add_argument('--video_driven_checkpoint', type=str, default='checkpoints/video_driven/video_driven.pth')
        self.add_argument('--pirender_checkpoint', type=str, default='checkpoints/pirender/checkpoint.pt')

        self.add_argument('--img_path', type=str, default='demo/video_driven/source/source_image/test.jpg')
        self.add_argument('--wav_path', type=str, default='demo/video_driven/source/audio/test.wav')
        self.add_argument('--video_3DMM_path', type=str, default='demo/video_driven/source/source_video_3DMM/test.npy')
        self.add_argument('--style_path', type=str, default='demo/video_driven/source/style_clip_3DMM/test.npy')
        self.add_argument('--save_path', type=str, default='demo/video_driven/results/test.mp4')



if __name__ == '__main__':
    args = ArgParserTest().parse_args()
    main(args)
