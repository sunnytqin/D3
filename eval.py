import time
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset
from decoder import TinyVAE, VanillaVAE
import wandb
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# def evaluate_synset(it_eval, net, latent_train, labels_train, testloader, args, return_loss=False, texture=False):
#     net = net.to(args.device)
#     latent_train = latent_train.to(args.device)
#     labels_train = labels_train.to(args.device)
#     lr = float(args.lr_net)
#     Epoch = int(args.epoch_eval_train)
#     lr_schedule = [Epoch//2+1]
#     # args.dsa_param.eval()
#     if args.optimizer_type == "SGD":
#         optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
#     elif args.optimizer_type == "Adam":
#         optimizer = torch.optim.Adam(net.parameters(), lr=lr)  

#     criterion = nn.CrossEntropyLoss().to(args.device)

#     dst_train = TensorDataset(latent_train, labels_train)
#     trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

#     start = time.time()
#     acc_train_list = []
#     loss_train_list = []
#     acc_test = -1.0

#     for ep in tqdm(range(Epoch+1)):
#         loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=(not args.offline_generate), texture=texture)
#         acc_train_list.append(acc_train)
#         loss_train_list.append(loss_train)
#         # if ep == Epoch:
#         if ep % 200 == 0:
#             with torch.no_grad():
#                 loss_test, acc_test_epoch = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
#                 if args.num_eval ==1: print(f"train loss: {loss_train:.2f}, train acc: {acc_train:.4f}, test acc: {acc_test_epoch:.4f}")
#                 if acc_test_epoch > acc_test:
#                     acc_test = acc_test_epoch
#         if ep in lr_schedule:
#             lr *= 0.1
#             if optimizer == "SGD":
#                 optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#             elif optimizer == "Adam":
#                 optimizer = torch.optim.Adam(net.parameters(), lr=lr)  

#     time_train = time.time() - start

#     print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

#     if return_loss:
#         return net, acc_train_list, acc_test, loss_train_list, loss_test
#     else:
#         return net, acc_train_list, acc_test
    

def main(args):

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    print("test size: ", len(dst_test))
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.eval_mode)

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        args.dc_aug_param = None


    distill_data = torch.load(f'checkpoints/{args.checkpoint_name}')
    kernel_num = distill_data['kernel_num']
    z_dim =  distill_data['z_dim']
    ipc = args.ipc
    print("ipc:", ipc)
    n_prototypes = num_classes * ipc
    print("n_prototypes", n_prototypes)

    latent = torch.randn(size=(n_prototypes, z_dim, 2))

    if args.decoder_size == 'small': # small decoder
        decoder = TinyVAE("small", im_size[0], channel, kernel_num=args.kernel_num, z_size=args.z_dim) 
    elif args.decoder_size == 'medium':
        if args.dataset in ["CIFAR100", "CIFAR10"]: # lower image resolution (32 X 32)
            decoder = VanillaVAE(channel, latent_dim=z_dim, linear_decode=args.linear_decode, hidden_dims=[64, 128, 256, 512])
        elif args.dataset in ["Tiny", "ImageNet64"]: # higher image resolution (64 X 64)
            decoder = VanillaVAE(channel, latent_dim=z_dim, linear_decode=args.linear_decode)
    elif args.decoder_size == 'large': # higher image resolution (64 X 64) only
        decoder = VanillaVAE(channel, latent_dim=args.z_dim, linear_decode=args.linear_decode, hidden_dims=[64, 128, 256, 512, 1024])
    else:
        raise ValueError('Unknown decoder size')


    decoder.load_state_dict(distill_data['vae_state_dict'])
    latent = distill_data['syn_img']
    
    label_syn = torch.tensor(np.array([np.ones(ipc, dtype=np.int_)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    decoder.to(args.device)
    latent.to(args.device)

    decoder.eval()

    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(
        sync_tensorboard=False,
        project="project_name",  
        entity="entity", 
        config=args,
        mode="disabled",
        )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.decoder = decoder
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    print('Evaluation model pool: ', model_eval_pool)


    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))

        accs_test = []
        accs_train = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

            eval_labs = label_syn
            with torch.no_grad():
                latent_save = latent
            
            latent_syn_eval, label_syn_eval = copy.deepcopy(latent_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, latent_syn_eval, label_syn_eval, testloader, args, texture=args.texture)

            accs_test.append(acc_test)
            accs_train.append(acc_train)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std
            save_this_it = True
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--optimizer_type', type=str, default='SGD', help='optimizer: SGD or Adam')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_aug', type=float, default=1e-02, help='learning rate for updating augmentation')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_net', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--decoder_size', type=str, default='small', choices=["small", "medium", "large"],
                        help='decoder size, see decoder.py for more info')
    parser.add_argument('--z_dim', type=int, default=64, help='hidden dimension for prototypes')
    parser.add_argument('--kernel_num', type=int, default=128, help='latent dimension for prototypes')
    parser.add_argument('--linear_decode', action=argparse.BooleanOptionalAction, default=True, help='use a linear layer to decode')
    
    parser.add_argument('--load_checkpoint', action='store_true', help="load pretrained VAE")

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--soft_label', action='store_true', help="not used here")
    parser.add_argument('--forward_fixed', action='store_true', help="VAE fixed generation (discrete case)")

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--expt_name', type=str, default=None, help='final model')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='checkpoint')
    

    args = parser.parse_args()

    main(args)


