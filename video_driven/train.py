import os, sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
from video_driven.model import *
from datasets.video_driven_dataset import StyleDataset_video_driven
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_preprocess.Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
import glob
torch.autograd.set_detect_anomaly(True)
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
    tag = config['tag']
    pipeline = config['pipeline']

    fileName_list = sorted(glob.glob1(config['model_path'],'*.pth'))
    if len(fileName_list) == 0:
        load_path = None
    else:
        load_path = os.path.join(config['model_path'], fileName_list[-1])
    
    style_encoder, _, _ = setup_style_encoder(args, config,
                                            version=None, load_path=None)
    


    style_discriminator3, _, _ = setup_style_discriminator(args, config,
                                            version=None, load_path=None)
    checkpoints = torch.load(args.style_encoder_checkpoint)
    style_encoder.load_state_dict( multi2single(checkpoints['state_dict']), strict=False)
    style_discriminator3.load_state_dict( multi2single(checkpoints['state_dict']), strict=False)


    style_discriminator = StyleDiscriminator(num_styles=config['style_class']).cuda()
    style_discriminator2 = Style_Discriminator2(config).cuda()
    ccc = torch.load(args.style_discriminator_checkpoint)
    style_discriminator.load_state_dict(multi2single(ccc['style_discriminator']))

    start_epoch = 0
    style_encoder = style_encoder.cuda()
    encoder = Encoder(args).cuda()

    ra = ResidualAdapter(args).cuda()

    decoder = Decoder(args).cuda()

    tem_discriminator = TemporalDiscriminator2().cuda()

    facemodel = ParametricFaceModel(
        bfm_folder='Deep3DFaceRecon_pytorch/BFM', camera_distance=10.0, focal=1015.0, center=112.0,
        is_train=False, default_name='BFM_model_front.mat'
    )

    facemodel.to('cuda')

    all_optimizer = torch.optim.Adam(list(encoder.parameters())+list(ra.parameters())+list(decoder.parameters()), lr=2e-4)
    tem_discriminator_optimizer = torch.optim.Adam(tem_discriminator.parameters(), lr=2e-4)
    style_discriminator2_optimizer = torch.optim.Adam(style_discriminator2.parameters(), lr=2e-4)

    if load_path:
        checkpoints = torch.load(load_path)
        encoder.load_state_dict(multi2single(checkpoints['encoder']))
        style_encoder.load_state_dict(multi2single(checkpoints['style_encoder']))
        ra.load_state_dict(multi2single(checkpoints['ra']))
        decoder.load_state_dict(multi2single(checkpoints['decoder']))
        tem_discriminator.load_state_dict(multi2single(checkpoints['tem_discriminator']))
        style_discriminator2.load_state_dict(multi2single(checkpoints['style_discriminator2']))

        all_optimizer.load_state_dict(checkpoints['optimizer']['all_optimizer'])
        tem_discriminator_optimizer.load_state_dict(checkpoints['optimizer']['tem_discriminator_optimizer'])
        style_discriminator2_optimizer.load_state_dict(checkpoints['optimizer']['style_discriminator2_optimizer'])
        start_epoch = checkpoints['epoch']+1
    print('model loaded!')

    encoder=torch.nn.parallel.DistributedDataParallel(encoder.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    style_encoder=torch.nn.parallel.DistributedDataParallel(style_encoder.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    ra=torch.nn.parallel.DistributedDataParallel(ra.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    decoder=torch.nn.parallel.DistributedDataParallel(decoder.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    tem_discriminator=torch.nn.parallel.DistributedDataParallel(tem_discriminator.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    # facemodel=torch.nn.parallel.DistributedDataParallel(facemodel.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    style_discriminator2=torch.nn.parallel.DistributedDataParallel(style_discriminator2.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)
    style_discriminator3=torch.nn.parallel.DistributedDataParallel(style_discriminator3.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)

    style_discriminator=torch.nn.parallel.DistributedDataParallel(style_discriminator.cuda(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=True)


    return config, encoder, style_encoder, ra, decoder, tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,facemodel,tem_discriminator_optimizer, all_optimizer,start_epoch



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def train_discriminator(config, bi, encoder, style_encoder, ra, decoder,tem_discriminator, style_discriminator):
    parameters, label, style_parameters, style_label, _, _ = bi
    l2_criterion = nn.MSELoss()

    requires_grad(encoder, False)
    requires_grad(style_encoder, False)
    requires_grad(ra, False)
    requires_grad(decoder, False)

    requires_grad(tem_discriminator, True)
    requires_grad(style_discriminator, True)


    parameters = parameters.type(torch.FloatTensor).cuda()
    label = label.type(torch.LongTensor).cuda()
    style_parameters = style_parameters.type(torch.FloatTensor).cuda()
    style_label = style_label.type(torch.LongTensor).cuda()

    source_style_code = style_encoder(parameters[:,:,80:144])
    target_style_code = style_encoder(style_parameters[:,:,80:144])
    z_t = encoder(parameters[:,:,80:144], source_style_code)
    z_t_prime, z_t_prime_recon = ra(z_t, target_style_code, source_style_code)
    results = decoder(z_t_prime, target_style_code)
    results_recon = decoder(z_t_prime_recon, source_style_code)

    fake_tem_socre1 = tem_discriminator(results)
    fake_tem_socre2 = tem_discriminator(results_recon)

    real_tem_socre1 = tem_discriminator(parameters[:,:,80:144])
    real_tem_socre2 = tem_discriminator(style_parameters[:,:,80:144])

    real_loss = 0.5 * l2_criterion(real_tem_socre1, torch.ones_like(real_tem_socre1))+0.5 * l2_criterion(real_tem_socre2, torch.ones_like(real_tem_socre2))
    fake_loss = 0.5 * l2_criterion(fake_tem_socre1, torch.zeros_like(real_tem_socre1))+0.5 * l2_criterion(fake_tem_socre2, torch.zeros_like(real_tem_socre2))

    tem_loss = 0.5*real_loss+0.5*fake_loss


    source_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    source_onehot.scatter_(1, label.view(-1, 1), 1)
    target_onehot = torch.zeros((label.shape[0], config['style_class'])).cuda()
    target_onehot.scatter_(1, style_label.view(-1, 1), 1)

    real_style_socre1, real_gp1 = style_discriminator(parameters[:,:,80:144], source_onehot, True)
    real_style_socre2, real_gp2 = style_discriminator(style_parameters[:,:,80:144], target_onehot, True)

    fake_style_socre1, _ = style_discriminator(results, target_onehot, False)
    fake_style_socre2, _ = style_discriminator(results_recon, source_onehot, False)

    real_loss = l2_criterion(real_style_socre1, torch.ones_like(real_style_socre1)) + l2_criterion(real_style_socre2, torch.ones_like(real_style_socre2))
    fake_loss = l2_criterion(fake_style_socre1, -torch.ones_like(fake_style_socre1)) + l2_criterion(fake_style_socre2, -torch.ones_like(fake_style_socre2))
    

    discrimitor_dict = {}
    discrimitor_dict['tem_loss'] = tem_loss
    discrimitor_dict['style_loss'] = 0.5*real_loss + 0.5*fake_loss
    discrimitor_dict['grad_loss'] = real_gp1*0.5 * 10 + real_gp2*0.5 * 10
    return discrimitor_dict


def gen(config, encoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
        cross_criterion, parameters, label, style_parameters, style_label, epoch, positive_parameters,negetive_parameters):


    source_style_code = style_encoder(parameters[:,:,80:144])
    target_style_code = style_encoder(style_parameters[:,:,80:144])

    z_t = encoder(parameters[:,:,80:144], source_style_code)
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
    recon_loss += config['theta']*l2_criterion(results_recon, parameters[:,:,80:144])

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

    cross_loss = 0.5*cross_criterion(result_class, style_label) + 0.5*cross_criterion(result_rec_class, label)

    fake_loss = l2_criterion(fake_style_socre1, torch.ones_like(fake_style_socre1)) + l2_criterion(fake_style_socre2, torch.ones_like(fake_style_socre2))
    
    result_z_t = encoder(results, target_style_code)
    result_z_t_prime = ra(result_z_t, source_style_code)
    result_cycle = decoder(result_z_t_prime, source_style_code)
    result_recon_loss = 0
    result_recon_loss += config['theta']* l2_criterion(result_cycle, parameters[:,:,80:144])

    result_recon_loss_2 = 0
    for i in range(32):
        tmp = parameters[:,i].clone().detach()
        tmp[:,80:144] = result_cycle[:,i]

        result_cycle_dict = facemodel.split_coeff(tmp)
        result_cycle_shape = facemodel.compute_shape(result_cycle_dict['id'], result_cycle_dict['exp'])

        result_recon_loss_2 += F.mse_loss(result_cycle_shape, cur_shapes[i], facemodel)

    result_recon_loss_2 = result_recon_loss_2/32
    result_recon_loss += result_recon_loss_2

    loss_dict = {}
    # if epoch<5:
    #     w_tem = 0
    # else:
    #     w_tem = config['loss_weights']['tem_loss']
    loss_dict['tem_loss'] = tem_loss*config['loss_weights']['tem_loss']
    loss_dict['mouth_loss'] = mouth_loss*config['loss_weights']['mouth_loss']
    loss_dict['recon_loss'] = recon_loss *config['loss_weights']['recon_loss']
    loss_dict['result_recon_loss'] = result_recon_loss*config['loss_weights']['result_recon_loss']
    loss_dict['dis_loss'] = fake_loss*config['loss_weights']['dis_loss']
    loss_dict['cross_loss'] = cross_loss*config['loss_weights']['cross_loss']
    loss_dict['dis_loss3'] = trip_loss2*config['loss_weights']['dis_loss3']

    # print(loss_dict)
    loss_values = [val.mean() for val in loss_dict.values()]
    loss = sum(loss_values)

    return loss, loss_dict, acc


def gen_3(config, encoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
        cross_criterion, parameters, label, style_parameters, style_label, epoch, positive_parameters,negetive_parameters):


    source_style_code = style_encoder(parameters[:,:,80:144])
    target_style_code = style_encoder(style_parameters[:,:,80:144])

    z_t = encoder(parameters[:,:,80:144], source_style_code)
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
    recon_loss += config['theta']*l2_criterion(results_recon, parameters[:,:,80:144])

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

    cross_loss = 0.5*cross_criterion(result_class, style_label) + 0.5*cross_criterion(result_rec_class, label)

    fake_loss = l2_criterion(fake_style_socre1, torch.ones_like(fake_style_socre1)) + l2_criterion(fake_style_socre2, torch.ones_like(fake_style_socre2))
    
    result_z_t = encoder(results, target_style_code)
    result_z_t_prime = ra(result_z_t, source_style_code)
    result_cycle = decoder(result_z_t_prime, source_style_code)
    result_recon_loss = 0
    result_recon_loss += config['theta']* l2_criterion(result_cycle, parameters[:,:,80:144])

    result_recon_loss_2 = 0
    for i in range(32):
        tmp = parameters[:,i].clone().detach()
        tmp[:,80:144] = result_cycle[:,i]

        result_cycle_dict = facemodel.split_coeff(tmp)
        result_cycle_shape = facemodel.compute_shape(result_cycle_dict['id'], result_cycle_dict['exp'])

        result_recon_loss_2 += F.mse_loss(result_cycle_shape, cur_shapes[i], facemodel)

    result_recon_loss_2 = result_recon_loss_2/32
    result_recon_loss += result_recon_loss_2

    loss_dict = {}

    loss_dict['tem_loss'] = tem_loss*config['loss_weights']['tem_loss']
    loss_dict['mouth_loss'] = mouth_loss*config['loss_weights']['mouth_loss']
    loss_dict['recon_loss'] = recon_loss *config['loss_weights']['recon_loss']
    loss_dict['result_recon_loss'] = result_recon_loss*config['loss_weights']['result_recon_loss']
    loss_dict['dis_loss'] = fake_loss*config['loss_weights']['dis_loss']
    loss_dict['cross_loss'] = cross_loss*config['loss_weights']['cross_loss']
    loss_dict['dis_loss3'] = trip_loss2*config['loss_weights']['dis_loss3']

    loss_values = [val.mean() for val in loss_dict.values()]
    loss = sum(loss_values)

    return loss, loss_dict, acc

def gen_2(config, encoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
        cross_criterion, parameters, label, style_parameters, style_label, epoch, positive_parameters,negetive_parameters):


    source_style_code = style_encoder(parameters[:,:,80:144])
    target_style_code = style_encoder(style_parameters[:,:,80:144])

    z_t = encoder(parameters[:,:,80:144], source_style_code)
    z_t_prime, z_t_prime_recon = ra(z_t, target_style_code, source_style_code)
    results = decoder(z_t_prime, target_style_code)
    results_recon = decoder(z_t_prime_recon, source_style_code)


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
    recon_loss += config['theta']*l2_criterion(results_recon, parameters[:,:,80:144])

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
    cross_loss = 0.5*cross_criterion(result_class, style_label) + 0.5*cross_criterion(result_rec_class, label)

    fake_loss = l2_criterion(fake_style_socre1, torch.ones_like(fake_style_socre1)) + l2_criterion(fake_style_socre2, torch.ones_like(fake_style_socre2))
    
    result_z_t = encoder(results, target_style_code)
    result_z_t_prime = ra(result_z_t, source_style_code)
    result_cycle = decoder(result_z_t_prime, source_style_code)
    result_recon_loss = 0
    result_recon_loss += config['theta']* l2_criterion(result_cycle, parameters[:,:,80:144])

    result_recon_loss_2 = 0
    for i in range(32):
        tmp = parameters[:,i].clone().detach()
        tmp[:,80:144] = result_cycle[:,i]

        result_cycle_dict = facemodel.split_coeff(tmp)
        result_cycle_shape = facemodel.compute_shape(result_cycle_dict['id'], result_cycle_dict['exp'])

        result_recon_loss_2 += F.mse_loss(result_cycle_shape, cur_shapes[i], facemodel)

    result_recon_loss_2 = result_recon_loss_2/32
    result_recon_loss += result_recon_loss_2

    loss_dict = {}

    loss_dict['tem_loss'] = tem_loss*config['loss_weights']['tem_loss']
    loss_dict['mouth_loss'] = mouth_loss*config['loss_weights']['mouth_loss']
    loss_dict['recon_loss'] = recon_loss *config['loss_weights']['recon_loss']
    loss_dict['result_recon_loss'] = result_recon_loss*config['loss_weights']['result_recon_loss']
    loss_dict['dis_loss'] = fake_loss*config['loss_weights']['dis_loss']
    loss_dict['cross_loss'] = cross_loss*config['loss_weights']['cross_loss']


    loss_values = [val.mean() for val in loss_dict.values()]
    loss = sum(loss_values)

    return loss


def calu_mouth_loss(shape_a, shape_b, bfm):
    lip_points = [51,57,62,66]
    lip_points_id = bfm.keypoints[lip_points]

    lip_diff_a = shape_a[:, lip_points_id[ ::2]] - shape_a[:, lip_points_id[1::2]]
    lip_diff_a = torch.sum(lip_diff_a**2, dim=2)

    lip_diff_b = shape_b[:, lip_points_id[ ::2]] - shape_b[:, lip_points_id[1::2]]
    lip_diff_b = torch.sum(lip_diff_b**2, dim=2)

    return F.l1_loss(lip_diff_a, lip_diff_b)

def generator_train_step(config, epoch, encoder, style_encoder, ra, decoder, all_optimizer,tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, dataloader, writer):
    totalSteps = len(dataloader)
    avgLoss = 0
    tem_avgLoss = 0
    recon_avgLoss = 0
    result_recon_avgLoss = 0
    cross_avgLoss = 0
    mouth_avgLoss = 0
    dis_avgLoss = 0
    trip_avgLoss = 0
    dis3_avgLoss = 0

    acc_total = 0


    dis_gen_avgLoss = 0

    batch_size = config['batch_size']
    encoder.train()
    style_encoder.train()
    ra.train()
    decoder.train()
    tem_discriminator.train()
    style_discriminator2.train()

    count = 0
    requires_grad(style_discriminator3, False)
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    cross_criterion = nn.CrossEntropyLoss()
    for bii, bi in enumerate(dataloader):
        if config['train_discriminator']:
            discriminator_loss = train_discriminator(config, bi, encoder, style_encoder, ra, decoder,tem_discriminator, style_discriminator2)
            style_discriminator2_optimizer.zero_grad()
            tem_discriminator_optimizer.zero_grad()

            dis_loss_values = [val for val in discriminator_loss.values()]
            dis_loss = sum(dis_loss_values)
            dis_loss.backward()
            style_discriminator2_optimizer.step()
            tem_discriminator_optimizer.step()
            dis_avgLoss += dis_loss.detach().item()
            requires_grad(encoder, True)
            requires_grad(style_encoder, False)
            requires_grad(ra, True)
            requires_grad(decoder, True)

            requires_grad(tem_discriminator, False)
            requires_grad(style_discriminator2, False)

        parameters, label,style_parameters, style_label,  positive_parameters, negetive_parameters = bi

        parameters = parameters.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()
        style_label = style_label.type(torch.LongTensor).cuda()


        positive_parameters = positive_parameters.type(torch.FloatTensor).cuda()
        negetive_parameters = negetive_parameters.type(torch.FloatTensor).cuda()

        l1, loss_dict, acc = gen(config, encoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                cross_criterion, parameters, label, style_parameters, style_label, epoch, positive_parameters,negetive_parameters)


        l2,_,_ = gen_3(config, encoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                cross_criterion, style_parameters, style_label, parameters, label, epoch, positive_parameters,negetive_parameters)
    
        loss = l1+l2

        acc_total += acc



        all_optimizer.zero_grad()

        loss.backward()

        all_optimizer.step()

        avgLoss += loss.detach().item()
        tem_avgLoss += loss_dict['tem_loss'].detach().item()
        mouth_avgLoss += loss_dict['mouth_loss'].detach().item()
        recon_avgLoss += loss_dict['recon_loss'].detach().item()
        result_recon_avgLoss += loss_dict['result_recon_loss'].detach().item()
        cross_avgLoss += loss_dict['cross_loss'].detach().item()
        dis_gen_avgLoss += loss_dict['dis_loss'].detach().item()
        dis3_avgLoss += loss_dict['dis_loss3'].detach().item()
        count += 1

        
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},acc:{:.4f}, tem_Loss: {:.4f},recon_Loss: {:.4f},result_recon_Loss: {:.4f},cross_Loss: {:.4f}, mouth_Loss: {:.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            avgLoss / count, acc, tem_avgLoss/ count, recon_avgLoss/ count, result_recon_avgLoss/ count, cross_avgLoss/ count, mouth_avgLoss/ count))

    writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)
    writer.add_scalar('mouth_loss/train_totalLoss', mouth_avgLoss / totalSteps, epoch)
    writer.add_scalar('tem_Loss/train_totalLoss', tem_avgLoss / totalSteps, epoch)
    writer.add_scalar('recon_Loss/train_totalLoss', recon_avgLoss / totalSteps, epoch)
    writer.add_scalar('result_recon_Loss/train_totalLoss', result_recon_avgLoss / totalSteps, epoch)
    writer.add_scalar('cross_Loss/train_totalLoss', cross_avgLoss / totalSteps, epoch)
    writer.add_scalar('dis_Loss/train_totalLoss', dis_avgLoss / totalSteps, epoch)
    writer.add_scalar('dis_gen_Loss/train_totalLoss', dis_gen_avgLoss / totalSteps, epoch)
    writer.add_scalar('acc/train_totalLoss', acc_total / totalSteps, epoch)
    writer.add_scalar('dis_Loss3/train_totalLoss', dis3_avgLoss / totalSteps, epoch)



def generator_val_step(config, epoch, encoder, style_encoder, ra, decoder, all_optimizer,tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, test_dataloader,
                               currBestLoss, prev_save_epoch, tag, writer):

    encoder.eval()
    style_encoder.eval()
    ra.eval()
    decoder.eval()
    tem_discriminator.eval()
    style_discriminator.eval()

    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    cross_criterion = nn.CrossEntropyLoss()

    totalSteps = len(test_dataloader)

    testLoss = 0
    tem_avgLoss = 0
    recon_avgLoss = 0
    result_recon_avgLoss = 0
    cross_avgLoss = 0
    mouth_avgLoss = 0
    dis_gen_avgLoss = 0
    trip_avgLoss = 0
    dis3_avgLoss = 0

    acc_total = 0

    for bii, bi in enumerate(test_dataloader):
        parameters, label,style_parameters, style_label,  positive_parameters, negetive_parameters = bi

        parameters = parameters.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()
        style_label = style_label.type(torch.LongTensor).cuda()

        positive_parameters = positive_parameters.type(torch.FloatTensor).cuda()
        negetive_parameters = negetive_parameters.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            l1, loss_dict, acc = gen(config, encoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                    cross_criterion, parameters, label, style_parameters, style_label, epoch, positive_parameters,negetive_parameters)


            l2,_,_ = gen_3(config, encoder, style_encoder, ra, decoder, facemodel, tem_discriminator, l1_criterion, style_discriminator, style_discriminator2, style_discriminator3, l2_criterion,\
                    cross_criterion, style_parameters, style_label, parameters, label, epoch, positive_parameters,negetive_parameters)
        
        loss = l1+l2

        acc_total += acc


        testLoss += loss.detach().item()

        tem_avgLoss += loss_dict['tem_loss'].detach().item()
        mouth_avgLoss += loss_dict['mouth_loss'].detach().item()
        recon_avgLoss += loss_dict['recon_loss'].detach().item()
        result_recon_avgLoss += loss_dict['result_recon_loss'].detach().item()
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
    writer.add_scalar('result_recon_Loss/val_totalLoss', result_recon_avgLoss / totalSteps, epoch)
    writer.add_scalar('cross_Loss/val_totalLoss', cross_avgLoss / totalSteps, epoch)
    writer.add_scalar('dis_gen_Loss/val_totalLoss', dis_gen_avgLoss / totalSteps, epoch)
    writer.add_scalar('acc/val_totalLoss', acc_total, epoch)
    writer.add_scalar('trip_loss/val_totalLoss', trip_avgLoss / totalSteps, epoch)
    writer.add_scalar('dis_Loss3/val_totalLoss', dis3_avgLoss / totalSteps, epoch)

    ## save model if curr loss is lower than previous best loss
    if testLoss < currBestLoss:
        prev_save_epoch = epoch #  encoder, style_encoder, ra, decoder, g_optimizer, all_optimizer,tem_discriminator, style_discriminator, style_discriminator_optimizer,tem_discriminator_optimizer
        checkpoint = {'config': args.config,
                        'encoder': encoder.state_dict(),
                        'style_encoder': style_encoder.state_dict(),
                        'ra': ra.state_dict(),
                        'decoder': decoder.state_dict(),
                        'tem_discriminator': tem_discriminator.state_dict(),
                        'style_discriminator': style_discriminator.state_dict(),
                        'style_discriminator2': style_discriminator2.state_dict(),
                        'style_discriminator3': style_discriminator3.state_dict(),
                        'optimizer': {
                        'all_optimizer': all_optimizer.state_dict(),
                        'tem_discriminator_optimizer': tem_discriminator_optimizer.state_dict(),
                        'style_discriminator2_optimizer': style_discriminator2_optimizer.state_dict(),
                        },
                        'epoch': epoch,
                        'currBestLoss':currBestLoss}
        os.makedirs(config['model_path'], exist_ok=True)
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
                            'encoder': encoder.state_dict(),
                            'style_encoder': style_encoder.state_dict(),
                            'ra': ra.state_dict(),
                            'decoder': decoder.state_dict(),
                            'tem_discriminator': tem_discriminator.state_dict(),
                            'style_discriminator': style_discriminator.state_dict(),
                            'style_discriminator2': style_discriminator2.state_dict(),
                            'optimizer': {
                            'all_optimizer': all_optimizer.state_dict(),
                            'tem_discriminator_optimizer': tem_discriminator_optimizer.state_dict(),
                            'style_discriminator_optimizer': style_discriminator2_optimizer.state_dict(),
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
    config, encoder, style_encoder, ra, decoder, tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,facemodel,tem_discriminator_optimizer, all_optimizer,start_epoch = load_model(args)
    encoder.train()
    style_encoder.eval()
    requires_grad(style_encoder, False)
    ra.train()
    decoder.train()
    tem_discriminator.train()
    style_discriminator2.train()


    tag = config['tag']
    pipeline = config['pipeline']


    currBestLoss = 1e3
    prev_save_epoch = 0
    
    writer = SummaryWriter('runs/debug_{}_{}'.format(tag, pipeline))

    dataset = StyleDataset_video_driven(is_train=True)
    test_dataset = StyleDataset_video_driven(is_train=False)

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
        generator_train_step(config, epoch, encoder, style_encoder, ra, decoder, all_optimizer, tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, dataloader, writer)
        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, encoder, style_encoder, ra, decoder, all_optimizer,tem_discriminator, style_discriminator, style_discriminator2, style_discriminator2_optimizer,style_discriminator3,tem_discriminator_optimizer, facemodel, test_dataloader,
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

        self.add_argument('--config', type=str, default='configs/video_driven/video_driven.json')
        self.add_argument('--style_discriminator_checkpoint', type=str, default=None)
        self.add_argument('--style_encoder_checkpoint', type=str, default=None)


if __name__ == '__main__':
    args = ArgParserTest().parse_args()
    init_dist(args.local_rank)
    main(args)
