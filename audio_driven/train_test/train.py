import os, sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
from audio_driven.models.model import *
from datasets.audio_driven_dataset import StyleDataset_audio_driven
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_preprocess.Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
import glob
from audio_driven.models.sdtw_cuda_loss import SoftDTW
import warnings
warnings.filterwarnings("ignore")
from distributed import init_dist
from distributed import master_only_print as print
from collections import OrderedDict

def multi2single(pretrain):
    new_state_dict = OrderedDict()
    for key, value in pretrain.items():
        name = key[7:]
        new_state_dict[name] = value
    return new_state_dict

def load_model(args):
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)


    load_path = None
    style_encoder, g_optimizer, start_epoch = setup_style_encoder(args, config,
                                            version=None, load_path=None)
    

    style_discriminator3, style_discriminator_optimizer, _ = setup_style_discriminator(args, config,
                                            version=None, load_path=None)


    checkpoints = torch.load(args.style_encoder_checkpoint)
    style_encoder.load_state_dict( multi2single(checkpoints['state_dict']), strict=False)
    style_discriminator3.load_state_dict( multi2single(checkpoints['state_dict']), strict=False)

    style_discriminator = StyleDiscriminator(num_styles=config['style_class']).cuda()
    style_discriminator2 = Style_Discriminator2(config).cuda()

    style_encoder = style_encoder.cuda()
    audioencoder = AudioEncoder2(args).cuda()

    ra = ResidualAdapter(args).cuda()

    decoder = Decoder(args).cuda()

    tem_discriminator = TemporalDiscriminator2().cuda()

    facemodel = ParametricFaceModel(
        bfm_folder='Deep3DFaceRecon_pytorch/BFM', camera_distance=10.0, focal=1015.0, center=112.0,
        is_train=False, default_name='BFM_model_front.mat'
    )

    audio_encoder_optimizer = torch.optim.Adam(audioencoder.parameters(), lr=2e-4)
    all_optimizer = torch.optim.Adam(list(ra.parameters())+list(decoder.parameters()), lr=2e-4)
    tem_discriminator_optimizer = torch.optim.Adam(tem_discriminator.parameters(), lr=2e-4)
    style_discriminator2_optimizer = torch.optim.Adam(style_discriminator2.parameters(), lr=2e-4)

    if load_path:
        checkpoints = torch.load(load_path)
        audioencoder.load_state_dict(multi2single(checkpoints['audio_encoder']))
        style_encoder.load_state_dict( multi2single(checkpoints['style_encoder']))
        ra.load_state_dict( multi2single(checkpoints['ra']))
        decoder.load_state_dict( multi2single(checkpoints['decoder']))
        tem_discriminator.load_state_dict( multi2single(checkpoints['tem_discriminator']))
        style_discriminator2.load_state_dict( multi2single(checkpoints['style_discriminator2']))
        style_discriminator.load_state_dict( multi2single(checkpoints['style_discriminator']))
        style_discriminator3.load_state_dict( multi2single(checkpoints['style_discriminator3']))
        audio_encoder_optimizer.load_state_dict(checkpoints['optimizer']['audio_encoder_optimizer'])
        all_optimizer.load_state_dict(checkpoints['optimizer']['all_optimizer'])
        tem_discriminator_optimizer.load_state_dict(checkpoints['optimizer']['tem_discriminator_optimizer'])
        style_discriminator2_optimizer.load_state_dict(checkpoints['optimizer']['style_discriminator2_optimizer'])
        currBestLoss = checkpoints['currBestLoss']
        start_epoch = checkpoints['epoch']+1
    print('model loaded!')


    audioencoder=torch.nn.parallel.DistributedDataParallel(audioencoder.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    style_encoder=torch.nn.parallel.DistributedDataParallel(style_encoder.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    ra=torch.nn.parallel.DistributedDataParallel(ra.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    decoder=torch.nn.parallel.DistributedDataParallel(decoder.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    tem_discriminator=torch.nn.parallel.DistributedDataParallel(tem_discriminator.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    # facemodel=torch.nn.parallel.DistributedDataParallel(facemodel.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    style_discriminator2=torch.nn.parallel.DistributedDataParallel(style_discriminator2.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    style_discriminator3=torch.nn.parallel.DistributedDataParallel(style_discriminator3.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)

    style_discriminator=torch.nn.parallel.DistributedDataParallel(style_discriminator.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)

    return config, audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer,  g_optimizer, tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,facemodel,tem_discriminator_optimizer, all_optimizer,start_epoch, currBestLoss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def train_discriminator(config, bi, audioencoder, style_encoder, ra, decoder,tem_discriminator, style_discriminator):
    parameters, label, audio_feature,example_parameters, style_parameters, style_label, style_audio_feature, style_example_parameters, positive_parameters, negetive_parameters = bi
    l2_criterion = nn.MSELoss()
    cross_criterion = nn.CrossEntropyLoss()
    
    requires_grad(audioencoder, False)
    requires_grad(style_encoder, False)
    requires_grad(ra, False)
    requires_grad(decoder, False)

    requires_grad(tem_discriminator, True)
    requires_grad(style_discriminator, True)

    parameters = parameters.type(torch.FloatTensor).cuda()
    label = label.type(torch.LongTensor).cuda()
    style_parameters = style_parameters.type(torch.FloatTensor).cuda()
    style_label = style_label.type(torch.LongTensor).cuda()

    example_parameters = example_parameters.type(torch.FloatTensor).cuda()
    audio_feature = audio_feature.type(torch.FloatTensor).cuda()

    style_example_parameters = style_example_parameters.type(torch.FloatTensor).cuda()
    style_audio_feature = style_audio_feature.type(torch.FloatTensor).cuda()


    source_style_code = style_encoder(parameters[:,:,80:144])
    target_style_code = style_encoder(style_parameters[:,:,80:144])


    z_t = audioencoder(audio_feature, example_parameters[:,80:144], source_style_code)
    z_t_prime, z_t_prime_recon = ra(z_t, target_style_code, source_style_code)
    results = decoder(z_t_prime, target_style_code)
    results_recon = decoder(z_t_prime_recon, source_style_code)

    z_t_n = audioencoder(style_audio_feature, style_example_parameters[:,80:144], target_style_code)
    z_t_prime_source, z_t_prime_style = ra(z_t_n, source_style_code, target_style_code)
    results_source = decoder(z_t_prime_source, source_style_code)
    results_style = decoder(z_t_prime_style, target_style_code)

    fake_tem_socre1 = tem_discriminator(results)
    fake_tem_socre2 = tem_discriminator(results_recon)
    fake_tem_socre3 = tem_discriminator(results_source)
    fake_tem_socre4 = tem_discriminator(results_style)

    real_tem_socre1 = tem_discriminator(parameters[:,:,80:144])
    real_tem_socre2 = tem_discriminator(style_parameters[:,:,80:144])

    real_loss = l2_criterion(real_tem_socre1, torch.ones_like(real_tem_socre1))+l2_criterion(real_tem_socre2, torch.ones_like(real_tem_socre2))
    fake_loss = 0.5 * l2_criterion(fake_tem_socre1, torch.zeros_like(real_tem_socre1))+0.5 * l2_criterion(fake_tem_socre2, torch.zeros_like(real_tem_socre2))\
    +0.5 * l2_criterion(fake_tem_socre3, torch.zeros_like(fake_tem_socre3))+0.5 * l2_criterion(fake_tem_socre4, torch.zeros_like(fake_tem_socre4))


    tem_loss = 0.5*real_loss+0.5*fake_loss


    source_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    source_onehot.scatter_(1, label.view(-1, 1), 1)
    target_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    target_onehot.scatter_(1, style_label.view(-1, 1), 1)


    real_style_socre1, real_gp1 = style_discriminator(parameters[:,:,80:144], source_onehot, True)
    real_style_socre2, real_gp2 = style_discriminator(style_parameters[:,:,80:144], target_onehot, True)

    fake_style_socre1, _ = style_discriminator(results, target_onehot, False)
    fake_style_socre2, _ = style_discriminator(results_recon, source_onehot, False)
    fake_style_socre3, _ = style_discriminator(results_source, source_onehot, False)
    fake_style_socre4, _ = style_discriminator(results_style, target_onehot, False)

    real_loss = l2_criterion(real_style_socre1, torch.ones_like(real_style_socre1)) + l2_criterion(real_style_socre2, torch.ones_like(real_style_socre2))
    fake_loss = l2_criterion(fake_style_socre1, -torch.ones_like(fake_style_socre1)) + l2_criterion(fake_style_socre2, -torch.ones_like(fake_style_socre2))\
    + l2_criterion(fake_style_socre3, -torch.ones_like(fake_style_socre3))\
        + l2_criterion(fake_style_socre4, -torch.ones_like(fake_style_socre4))
    

    discrimitor_dict = {}
    discrimitor_dict['tem_loss'] = tem_loss
    discrimitor_dict['style_loss'] = real_loss + 0.5*fake_loss
    discrimitor_dict['grad_loss'] = real_gp1*0.5 * 10 + real_gp2*0.5 * 10
    return discrimitor_dict


def cosine_similarity(tensor1, tensor2):
    # 计算张量的内积
    dot_product = torch.sum(tensor1 * tensor2, dim=1)
    # 计算张量的长度
    tensor1_norm = torch.norm(tensor1, dim=1)
    tensor2_norm = torch.norm(tensor2, dim=1)
    # 计算余弦相似度
    cosine_similarities = dot_product / (tensor1_norm * tensor2_norm)
    return cosine_similarities

# config, encoder, audioencoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                # cross_criterion, parameters, example_parameters, audioencoder,audio_feature, label, style_parameters, style_label, epoch, positive_parameters,negetive_parameters, s_negetive_parameters, s_negetive_audio_feature,s_negetive_label,dtw_criterion
def gen(config, audioencoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                cross_criterion, parameters, example_parameters,audio_feature, label, style_parameters, style_example_parameters, style_audio_feature, style_label, epoch, positive_parameters,negetive_parameters,dtw_criterion):

    facemodel.to('cuda')
    source_style_code = style_encoder(parameters[:,:,80:144])
    target_style_code = style_encoder(style_parameters[:,:,80:144])


    z_t = audioencoder(audio_feature, example_parameters[:,80:144], source_style_code)



    z_t_prime, z_t_prime_recon = ra(z_t, target_style_code, source_style_code)
    results = decoder(z_t_prime, target_style_code)
    results_recon = decoder(z_t_prime_recon, source_style_code)

    results_style_code = style_discriminator3(results)

    trip_loss2 = max(l2_criterion(results_style_code, style_discriminator3(style_parameters[:,:,80:144]))\
                     +l2_criterion(style_discriminator3(results_recon), style_discriminator3(parameters[:,:,80:144]))\
                            + l2_criterion(results_style_code, style_discriminator3(positive_parameters[:,:,80:144]))\
                                - l2_criterion(results_style_code, style_discriminator3(negetive_parameters[:,:,80:144]))\
                                    - l2_criterion(results_style_code, style_discriminator3(parameters[:,:,80:144]))+5,0)

    mouth_loss = 0
    recon_loss_2 = 0
    recon_loss = 0


    cur_shapes = []
    for i in range(0,32):
        tmp = parameters[:,i].clone().detach()
        tmp[:,80:144] = results[:,i]

        style_pred_dict = facemodel.split_coeff(tmp)
        style_pred_shape = facemodel.compute_shape(style_pred_dict['id'], style_pred_dict['exp'])
        
        cur = parameters[:,i].clone().detach()
        cur_dict = facemodel.split_coeff(cur)
        cur_shape = facemodel.compute_shape(cur_dict['id'], cur_dict['exp'])
        cur_shapes.append(cur_shape)

        mouth_loss += calu_mouth_loss(style_pred_shape, cur_shape, facemodel)

        tmp = parameters[:,i].clone().detach()
        tmp[:,80:144] = results_recon[:,i]
        rec_pred_dict = facemodel.split_coeff(tmp)
        rec_pred_shape = facemodel.compute_shape(rec_pred_dict['id'], rec_pred_dict['exp'])
        
        mouth_loss+= calu_mouth_loss(rec_pred_shape, cur_shape, facemodel)

        recon_loss_2 +=F.mse_loss(rec_pred_shape, cur_shape)

    mouth_loss = mouth_loss/32
    recon_loss_2 = recon_loss_2/32
    recon_loss += recon_loss_2
    recon_loss += dtw_criterion(results_recon[:,:,:18], parameters[:,:,80:98]).mean(0)


    tem_socre1 = tem_discriminator(results)
    tem_socre2 = tem_discriminator(results_recon)

    tem_loss = 0.5*l2_criterion(tem_socre1, torch.ones_like(tem_socre1)) + 0.5*l2_criterion(tem_socre2, torch.ones_like(tem_socre2))

    source_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    source_onehot.scatter_(1, label.view(-1, 1), 1)

    target_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    target_onehot.scatter_(1, style_label.view(-1, 1), 1)


    fake_style_socre1, _ = style_discriminator2(results, target_onehot, False)
    fake_style_socre2, _ = style_discriminator2(results_recon, source_onehot, False)

    result_class = style_discriminator(results)
    result_rec_class = style_discriminator(results_recon)

    _, predicted_ = torch.max(result_class.data, 1)
    _, predicted_2 = torch.max(result_rec_class.data, 1)
    acc = ((predicted_ == style_label).sum().item() + (predicted_2 == label).sum().item())/((style_label == style_label).sum().item() + (label == label).sum().item())
    # print('style: ', (predicted_ == style_label).sum().item(), 'recon: ', (predicted_2 == label).sum().item())
    cross_loss = 0.5*cross_criterion(result_class, style_label) + 0.5*cross_criterion(result_rec_class, label)

    fake_loss = l2_criterion(fake_style_socre1, torch.ones_like(fake_style_socre1)) + l2_criterion(fake_style_socre2, torch.ones_like(fake_style_socre2))

    loss_dict = {}

    loss_dict['tem_loss'] = tem_loss*config['loss_weights']['tem_loss']
    loss_dict['mouth_loss'] = mouth_loss*config['loss_weights']['mouth_loss']
    loss_dict['recon_loss'] = recon_loss *config['loss_weights']['recon_loss']
    # loss_dict['result_recon_loss'] = result_recon_loss*config['loss_weights']['result_recon_loss']
    loss_dict['dis_loss'] = fake_loss*config['loss_weights']['dis_loss']
    loss_dict['cross_loss'] = cross_loss*config['loss_weights']['cross_loss']
    loss_dict['dis_loss3'] = trip_loss2*config['loss_weights']['dis_loss3']

    # print(loss_dict)
    loss_values = [val.mean() for val in loss_dict.values()]
    loss = sum(loss_values)

    return loss, loss_dict, acc


def gen_2(config, audioencoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                cross_criterion, parameters, example_parameters,audio_feature, label, style_parameters, style_example_parameters, style_audio_feature, style_label, epoch, positive_parameters,negetive_parameters,dtw_criterion):

    facemodel.to('cuda')
    source_style_code = style_encoder(parameters[:,:,80:144])
    target_style_code = style_encoder(style_parameters[:,:,80:144])



    z_t = audioencoder(audio_feature, example_parameters[:,80:144], source_style_code)



    z_t_prime, z_t_prime_recon = ra(z_t, target_style_code, source_style_code)
    results = decoder(z_t_prime, target_style_code)
    results_recon = decoder(z_t_prime_recon, source_style_code)

    results_style_code = style_discriminator3(results)

    trip_loss2 = max(l2_criterion(results_style_code, style_discriminator3(style_parameters[:,:,80:144]))\
                     +l2_criterion(style_discriminator3(results_recon), style_discriminator3(parameters[:,:,80:144]))\
                            + l2_criterion(style_discriminator3(results_recon), style_discriminator3(positive_parameters[:,:,80:144]))\
                                - l2_criterion(style_discriminator3(results_recon), style_discriminator3(negetive_parameters[:,:,80:144]))\
                                    - l2_criterion(results_style_code, style_discriminator3(parameters[:,:,80:144]))+5,0)

    mouth_loss = 0
    recon_loss_2 = 0
    recon_loss = 0


    cur_shapes = []
    for i in range(0,32):
        tmp = parameters[:,i].clone().detach()
        tmp[:,80:144] = results[:,i]

        style_pred_dict = facemodel.split_coeff(tmp)
        style_pred_shape = facemodel.compute_shape(style_pred_dict['id'], style_pred_dict['exp'])
        
        cur = parameters[:,i].clone().detach()
        cur_dict = facemodel.split_coeff(cur)
        cur_shape = facemodel.compute_shape(cur_dict['id'], cur_dict['exp'])
        cur_shapes.append(cur_shape)

        mouth_loss += calu_mouth_loss(style_pred_shape, cur_shape, facemodel)

        tmp = parameters[:,i].clone().detach()
        tmp[:,80:144] = results_recon[:,i]
        rec_pred_dict = facemodel.split_coeff(tmp)
        rec_pred_shape = facemodel.compute_shape(rec_pred_dict['id'], rec_pred_dict['exp'])
        
        mouth_loss+= calu_mouth_loss(rec_pred_shape, cur_shape, facemodel)

        recon_loss_2 +=F.mse_loss(rec_pred_shape, cur_shape)

    mouth_loss = mouth_loss/32
    recon_loss_2 = recon_loss_2/32
    recon_loss += recon_loss_2
    recon_loss += dtw_criterion(results_recon[:,:,:18], parameters[:,:,80:98]).mean(0)


    tem_socre1 = tem_discriminator(results)
    tem_socre2 = tem_discriminator(results_recon)

    tem_loss = 0.5*l2_criterion(tem_socre1, torch.ones_like(tem_socre1)) + 0.5*l2_criterion(tem_socre2, torch.ones_like(tem_socre2))

    source_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    source_onehot.scatter_(1, label.view(-1, 1), 1)

    target_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    target_onehot.scatter_(1, style_label.view(-1, 1), 1)


    fake_style_socre1, _ = style_discriminator2(results, target_onehot, False)
    fake_style_socre2, _ = style_discriminator2(results_recon, source_onehot, False)

    result_class = style_discriminator(results)
    result_rec_class = style_discriminator(results_recon)

    _, predicted_ = torch.max(result_class.data, 1)
    _, predicted_2 = torch.max(result_rec_class.data, 1)
    acc = ((predicted_ == style_label).sum().item() + (predicted_2 == label).sum().item())/((style_label == style_label).sum().item() + (label == label).sum().item())
    # print('style: ', (predicted_ == style_label).sum().item(), 'recon: ', (predicted_2 == label).sum().item())
    cross_loss = 0.5*cross_criterion(result_class, style_label) + 0.5*cross_criterion(result_rec_class, label)

    fake_loss = l2_criterion(fake_style_socre1, torch.ones_like(fake_style_socre1)) + l2_criterion(fake_style_socre2, torch.ones_like(fake_style_socre2))

    loss_dict = {}

    loss_dict['tem_loss'] = tem_loss*config['loss_weights']['tem_loss']
    loss_dict['mouth_loss'] = mouth_loss*config['loss_weights']['mouth_loss']
    loss_dict['recon_loss'] = recon_loss *config['loss_weights']['recon_loss']
    # loss_dict['result_recon_loss'] = result_recon_loss*config['loss_weights']['result_recon_loss']
    loss_dict['dis_loss'] = fake_loss*config['loss_weights']['dis_loss']
    loss_dict['cross_loss'] = cross_loss*config['loss_weights']['cross_loss']
    loss_dict['dis_loss3'] = trip_loss2*config['loss_weights']['dis_loss3']

    # print(loss_dict)
    loss_values = [val.mean() for val in loss_dict.values()]
    loss = sum(loss_values)

    return loss, loss_dict, acc


def calu_mouth_loss(shape_a, shape_b, bfm):
    lip_points = [51,57,62,66]
    lip_points_id = bfm.keypoints[lip_points]

    lip_diff_a = shape_a[:, lip_points_id[ ::2]] - shape_a[:, lip_points_id[1::2]]
    lip_diff_a = torch.sum(lip_diff_a**2, dim=2)

    lip_diff_b = shape_b[:, lip_points_id[ ::2]] - shape_b[:, lip_points_id[1::2]]
    lip_diff_b = torch.sum(lip_diff_b**2, dim=2)

    return F.l1_loss(lip_diff_a, lip_diff_b)
    # config, epoch, encoder, audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer, g_optimizer, all_optimizer, tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, dataloader, writer
def generator_train_step(config, epoch, audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer, g_optimizer, all_optimizer,tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, dataloader, writer):
    totalSteps = len(dataloader)
    avgLoss = 0
    tem_avgLoss = 0
    recon_avgLoss = 0
    # result_recon_avgLoss = 0
    cross_avgLoss = 0
    mouth_avgLoss = 0
    dis_avgLoss = 0
    dis3_avgLoss = 0

    acc_total = 0


    dis_gen_avgLoss = 0

    batch_size = config['batch_size']
    audioencoder.train()
    # style_encoder.train()
    ra.train()
    decoder.train()
    tem_discriminator.train()
    style_discriminator2.train()

    count = 0
    requires_grad(style_discriminator3, False)
    requires_grad(style_discriminator, False)
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    dtw_criterion = SoftDTW(use_cuda=True, gamma=config['gamma'])
    cross_criterion = nn.CrossEntropyLoss()
    for bii, bi in enumerate(dataloader):
        if config['train_discriminator']:
            discriminator_loss = train_discriminator(config, bi, audioencoder, style_encoder, ra, decoder,tem_discriminator, style_discriminator2)
            style_discriminator2_optimizer.zero_grad()
            tem_discriminator_optimizer.zero_grad()

            dis_loss_values = [val for val in discriminator_loss.values()]
            dis_loss = sum(dis_loss_values)
            dis_loss.backward()
            style_discriminator2_optimizer.step()
            tem_discriminator_optimizer.step()
            dis_avgLoss += dis_loss.detach().item()

        

        requires_grad(audioencoder, True)
        requires_grad(style_encoder, False)
        requires_grad(ra, True)
        requires_grad(decoder, True)

        requires_grad(tem_discriminator, False)
        requires_grad(style_discriminator2, False)

        
        parameters, label, audio_feature,example_parameters, style_parameters, style_label, style_audio_feature, style_example_parameters, positive_parameters, negetive_parameters = bi
        
        parameters = parameters.type(torch.FloatTensor).cuda()
        example_parameters = example_parameters.type(torch.FloatTensor).cuda()
        audio_feature = audio_feature.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()
        style_label = style_label.type(torch.LongTensor).cuda()
        style_example_parameters = style_example_parameters.type(torch.FloatTensor).cuda()
        style_audio_feature = style_audio_feature.type(torch.FloatTensor).cuda()


        positive_parameters = positive_parameters.type(torch.FloatTensor).cuda()
        negetive_parameters = negetive_parameters.type(torch.FloatTensor).cuda()

        l1, loss_dict, acc = gen(config, audioencoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                cross_criterion, parameters, example_parameters,audio_feature, label, style_parameters, style_example_parameters, style_audio_feature, style_label, epoch, positive_parameters,negetive_parameters,dtw_criterion)

        l2, _, _ = gen(config, audioencoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                cross_criterion, style_parameters, style_example_parameters, style_audio_feature, style_label,parameters, example_parameters,audio_feature, label,  epoch, positive_parameters,negetive_parameters,dtw_criterion)

        loss = l1+l2

        acc_total += acc



        audio_encoder_optimizer.zero_grad()
        # g_optimizer.zero_grad()
        all_optimizer.zero_grad()

        loss.backward()
        audio_encoder_optimizer.step()
        # g_optimizer.step_and_update_lr()
        all_optimizer.step()




        avgLoss += loss.detach().item()
        tem_avgLoss += loss_dict['tem_loss'].detach().item()
        mouth_avgLoss += loss_dict['mouth_loss'].detach().item()
        recon_avgLoss += loss_dict['recon_loss'].detach().item()
        # result_recon_avgLoss += loss_dict['result_recon_loss'].detach().item()
        cross_avgLoss += loss_dict['cross_loss'].detach().item()
        dis_gen_avgLoss += loss_dict['dis_loss'].detach().item()
        dis3_avgLoss += loss_dict['dis_loss3'].detach().item()
        count += 1

        
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},acc:{:.4f}, tem_Loss: {:.4f},recon_Loss: {:.4f},cross_Loss: {:.4f}, mouth_Loss: {:.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            avgLoss / count, acc, tem_avgLoss/ count, recon_avgLoss/ count, cross_avgLoss/ count, mouth_avgLoss/ count))

    writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)
    writer.add_scalar('mouth_loss/train_totalLoss', mouth_avgLoss / totalSteps, epoch)
    writer.add_scalar('tem_Loss/train_totalLoss', tem_avgLoss / totalSteps, epoch)
    writer.add_scalar('recon_Loss/train_totalLoss', recon_avgLoss / totalSteps, epoch)
    # writer.add_scalar('result_recon_Loss/train_totalLoss', result_recon_avgLoss / totalSteps, epoch)
    writer.add_scalar('cross_Loss/train_totalLoss', cross_avgLoss / totalSteps, epoch)
    writer.add_scalar('dis_Loss/train_totalLoss', dis_avgLoss / totalSteps, epoch)
    writer.add_scalar('dis_gen_Loss/train_totalLoss', dis_gen_avgLoss / totalSteps, epoch)
    writer.add_scalar('acc/train_totalLoss', acc_total / totalSteps, epoch)
    writer.add_scalar('dis_Loss3/train_totalLoss', dis3_avgLoss / totalSteps, epoch)

# config, epoch, audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer, g_optimizer, all_optimizer,tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, test_dataloader,
#                                currBestLoss, prev_save_epoch, tag, writer
def generator_val_step(config, epoch,audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer, g_optimizer, all_optimizer,tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, test_dataloader,
                               currBestLoss, prev_save_epoch, tag, writer):

    style_encoder.eval()
    ra.eval()
    decoder.eval()
    tem_discriminator.eval()
    style_discriminator.eval()

    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    cross_criterion = nn.CrossEntropyLoss()
    dtw_criterion = SoftDTW(use_cuda=True, gamma=config['gamma'])
    totalSteps = len(test_dataloader)

    testLoss = 0
    tem_avgLoss = 0
    recon_avgLoss = 0
    result_recon_avgLoss = 0
    cross_avgLoss = 0
    mouth_avgLoss = 0
    dis_gen_avgLoss = 0
    dis3_avgLoss = 0

    acc_total = 0
    batch_size = config['batch_size']

    for bii, bi in enumerate(test_dataloader):
        parameters, label, audio_feature,example_parameters, style_parameters, style_label, style_audio_feature, style_example_parameters, positive_parameters, negetive_parameters = bi
        
        parameters = parameters.type(torch.FloatTensor).cuda()
        example_parameters = example_parameters.type(torch.FloatTensor).cuda()
        audio_feature = audio_feature.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()
        style_label = style_label.type(torch.LongTensor).cuda()
        style_example_parameters = style_example_parameters.type(torch.FloatTensor).cuda()
        style_audio_feature = style_audio_feature.type(torch.FloatTensor).cuda()


        positive_parameters = positive_parameters.type(torch.FloatTensor).cuda()
        negetive_parameters = negetive_parameters.type(torch.FloatTensor).cuda()

        with torch.no_grad():
            l1, loss_dict, acc =gen(config, audioencoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                    cross_criterion, parameters, example_parameters,audio_feature, label, style_parameters, style_example_parameters, style_audio_feature, style_label, epoch, positive_parameters,negetive_parameters,dtw_criterion)

        loss = l1

        acc_total += acc


        testLoss += loss.detach().item()

        tem_avgLoss += loss_dict['tem_loss'].detach().item()
        mouth_avgLoss += loss_dict['mouth_loss'].detach().item()
        recon_avgLoss += loss_dict['recon_loss'].detach().item()
        # result_recon_avgLoss += loss_dict['result_recon_loss'].detach().item()
        cross_avgLoss += loss_dict['cross_loss'].detach().item()
        dis_gen_avgLoss += loss_dict['dis_loss'].detach().item()
        dis3_avgLoss += loss_dict['dis_loss3'].detach().item()
        print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, acc:{:.4f}'\
            .format(epoch, config['num_epochs'], bii, totalSteps,
                    loss.detach().item(), acc))
    testLoss /= totalSteps
    acc_total /= totalSteps

    


    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss, epoch)
    writer.add_scalar('mouth_loss/val_totalLoss', mouth_avgLoss / totalSteps, epoch)
    writer.add_scalar('tem_Loss/val_totalLoss', tem_avgLoss / totalSteps, epoch)
    writer.add_scalar('recon_Loss/val_totalLoss', recon_avgLoss / totalSteps, epoch)
    # writer.add_scalar('result_recon_Loss/val_totalLoss', result_recon_avgLoss / totalSteps, epoch)
    writer.add_scalar('cross_Loss/val_totalLoss', cross_avgLoss / totalSteps, epoch)
    writer.add_scalar('dis_gen_Loss/val_totalLoss', dis_gen_avgLoss / totalSteps, epoch)
    writer.add_scalar('acc/val_totalLoss', acc_total, epoch)
    writer.add_scalar('dis_Loss3/val_totalLoss', dis3_avgLoss / totalSteps, epoch)

    ## save model if curr loss is lower than previous best loss
    if testLoss < currBestLoss:
        prev_save_epoch = epoch #  encoder, style_encoder, ra, decoder, g_optimizer, all_optimizer,tem_discriminator, style_discriminator, style_discriminator_optimizer,tem_discriminator_optimizer
        checkpoint = {'config': args.config,
                        'audio_encoder': audioencoder.state_dict(),
                        'style_encoder': style_encoder.state_dict(),
                        'ra': ra.state_dict(),
                        'decoder': decoder.state_dict(),
                        'tem_discriminator': tem_discriminator.state_dict(),
                        'style_discriminator': style_discriminator.state_dict(),
                        'style_discriminator2': style_discriminator2.state_dict(),
                        'style_discriminator3': style_discriminator3.state_dict(),
                        'optimizer': {
                        'audio_encoder_optimizer':audio_encoder_optimizer.state_dict(),
                        'g_optimizer': g_optimizer._optimizer.state_dict(),
                        'all_optimizer': all_optimizer.state_dict(),
                        'tem_discriminator_optimizer': tem_discriminator_optimizer.state_dict(),
                        'style_discriminator2_optimizer': style_discriminator2_optimizer.state_dict(),
                        'n_steps': g_optimizer.n_steps,
                        },
                        'epoch': epoch,
                        'currBestLoss':currBestLoss}
        os.makedirs(config['model_path'], exist_ok=True)
        fileName = config['model_path'] + \
                        '{}{}_best_{}.pth'.format(tag, config['pipeline'], '%04d'%epoch)
        print('>>>> saving best epoch {}'.format(epoch), testLoss)
        currBestLoss = testLoss
        # if epoch%config['save_epoch']==0:
        torch.save(checkpoint, fileName)
        return currBestLoss, prev_save_epoch, testLoss
    else:
        if epoch%config['save_epoch'] == 0:
            prev_save_epoch = epoch
            checkpoint = {'config': args.config,
                            'audio_encoder': audioencoder.state_dict(),
                            'style_encoder': style_encoder.state_dict(),
                            'ra': ra.state_dict(),
                            'decoder': decoder.state_dict(),
                            'tem_discriminator': tem_discriminator.state_dict(),
                            'style_discriminator': style_discriminator.state_dict(),
                            'style_discriminator2': style_discriminator2.state_dict(),
                            'style_discriminator3': style_discriminator3.state_dict(),
                            'optimizer': {
                            'audio_encoder_optimizer':audio_encoder_optimizer.state_dict(),
                            'g_optimizer': g_optimizer._optimizer.state_dict(),
                            'all_optimizer': all_optimizer.state_dict(),
                            'tem_discriminator_optimizer': tem_discriminator_optimizer.state_dict(),
                            'style_discriminator2_optimizer': style_discriminator2_optimizer.state_dict(),
                            'n_steps': g_optimizer.n_steps,
                            },
                            'epoch': epoch,
                        'currBestLoss':currBestLoss}
            os.makedirs(config['model_path'], exist_ok=True)
            fileName = config['model_path'] + \
                            '{}{}_no_best_{}.pth'.format(tag, config['pipeline'], '%04d'%epoch)
            
            torch.save(checkpoint, fileName)
        return currBestLoss, prev_save_epoch, currBestLoss
def main(args):
    """ full pipeline for training the Predictor model """
    config, audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer, g_optimizer, tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,facemodel,tem_discriminator_optimizer, all_optimizer,start_epoch, currBestLoss = load_model(args)
    audioencoder.train()
    style_encoder.eval()
    requires_grad(style_encoder, False)
    ra.train()
    decoder.train()
    tem_discriminator.train()
    style_discriminator2.train()


    tag = config['tag']
    pipeline = config['pipeline']


    # currBestLoss = 1e3
    start_epoch = start_epoch

    writer = SummaryWriter('runs/debug_{}_{}'.format(tag, pipeline))

    prev_save_epoch = 0
    dataset = StyleDataset_audio_driven(is_train=True)
    test_dataset = StyleDataset_audio_driven(is_train=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None), num_workers=4, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=(test_sampler is None), num_workers=4, sampler=test_sampler, pin_memory=True, drop_last=True)
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch+config['num_epochs']-1:
            print('early stopping at:', epoch)
            print('best loss:', currBestLoss)
            break
        dataloader.sampler.set_epoch(epoch)
        test_dataloader.sampler.set_epoch(epoch)
        generator_train_step(config, epoch, audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer, g_optimizer, all_optimizer, tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, dataloader, writer)
        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, audioencoder, style_encoder, ra, decoder,audio_encoder_optimizer, g_optimizer, all_optimizer,tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, test_dataloader,
                               currBestLoss, prev_save_epoch, tag, writer)
    print('final best loss:', currBestLoss)


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

        self.add_argument('--style_discriminator_checkpoint', type=str, default=None)
        self.add_argument('--style_encoder_checkpoint', type=str, default=None)

if __name__ == '__main__':


    args = ArgParserTest().parse_args()
    init_dist(args.local_rank)
    main(args)
