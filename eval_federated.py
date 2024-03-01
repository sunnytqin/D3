'''
Evaluate the performance of federated distillation 
'''
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from utils import get_dataset, get_network, get_eval_pool, evaluate_synset_batch, get_time, get_memory, count_parameters, DiffAugment, ParamDiffAug
from decoder import TinyVAE, VanillaVAE

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if args.dsa:
        args.dc_aug_param = None

    no_batches = args.num_subtasks # number of subtasks for federated distillation
    
    decoders = []
    for _ in range(no_batches):
        if args.decoder_size == 'small':
            decoders.append(TinyVAE("small", im_size[0], channel, kernel_num=args.kernel_num, z_size=args.z_dim)) # small decoder, used for 1IPC equivalent case
        elif args.decoder_size == 'medium':
            decoders.append(VanillaVAE(channel, latent_dim=args.z_dim)) # medium decoder, used for 2IPC equivalent case
        elif args.decoder_size == 'large':
            decoders.append(VanillaVAE(channel, latent_dim=args.z_dim, linear_decode=args.linear_decode, hidden_dims=[64, 128, 256, 512, 1024])) # large decoder, used for 10IPC equivalent case
        else:
            raise ValueError('Unknown decoder size')

    args.vae_batch = num_classes // len(decoders)
    
    # send to GPU
    for d in decoders:
        d.to(torch.device("cuda"))

    args.decoder = decoders

    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(
        sync_tensorboard=False,
        project="project_name",  
        entity="entity", 
        config=args,
        name=args.run_name,
        notes=args.notes,
        mode="disabled",
        )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.decoder = decoders
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.lpc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Decoder size: ', count_parameters(args.decoder[0]))
    print('Evaluation model pool: ', model_eval_pool)

   
    ''' initialize the synthetic data '''
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    image_syn = []
    label_syn = []
    for batch_no in range(no_batches):
        image_syn.append(torch.randn(size=(args.vae_batch * args.lpc, args.z_dim, 2), requires_grad=False, device=args.device))
        label_syn.append(torch.tensor(batch_no * args.vae_batch + np.array([np.ones(args.lpc, dtype=np.int_)*i for i in range(args.vae_batch)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1)) # [0,0,0, 1,1,1, ..., 9,9,9]

    ''' load checkpoint - please use the corresponding part based on the decoder '''
    # this part of the code is obviously written tailored to current three distilled data sizes. modify accordingly if you are customizing the pipeline
    # 1 IPC equivalent - "small"
    if args.lpc == 1: # 1 IPC equivalent - "small", also uses 1 lpc
        checkpoint_names = ['ImageNet64_imagebatch_0_ipc1_final.pt',
                            'ImageNet64_imagebatch_1_ipc1_final.pt',
                            'ImageNet64_imagebatch_2_ipc1_final.pt',
                            'ImageNet64_imagebatch_3_ipc1_final.pt',
                            'ImageNet64_imagebatch_4_ipc1_final.pt',] 
        
        softlabel_names = ['ImageNet64_imagebatch_0_ipc1_softlabel.pt',
                            'ImageNet64_imagebatch_1_ipc1_softlabel.pt',
                            'ImageNet64_imagebatch_2_ipc1_softlabel.pt',
                            'ImageNet64_imagebatch_3_ipc1_softlabel.pt',
                            'ImageNet64_imagebatch_4_ipc1_softlabel.pt',]

    elif args.lpc == 2: # 2 IPC equivalent - "medium", also uses 2 lpc 
        checkpoint_names = ['ImageNet64_imagenethalf_0_ipc2_final.pt',
                            'ImageNet64_imagenethalf_1_ipc2_final.pt']
        
        softlabel_names = ['ImageNet64_imagenethalf_0_ipc2_softlabel.pt',
                            'ImageNet64_imagenethalf_1_ipc2_softlabel.pt']
        
    elif args.lpc == 10: # 10 IPC equivalent - "large", also uses 10 lpc
        checkpoint_names = ['ImageNet64_imagebatch_0_ipc10_large_decoder_nolinear_final.pt',
                            'ImageNet64_imagebatch_1_ipc10_large_decoder_nolinear_final.pt',
                            'ImageNet64_imagebatch_2_ipc10_large_decoder_nolinear_final.pt',
                            'ImageNet64_imagebatch_3_ipc10_large_decoder_nolinear_final.pt',
                            'ImageNet64_imagebatch_4_ipc10_large_decoder_nolinear_final.pt',
                            ]
        softlabel_names = ['ImageNet64_imagebatch_0_ipc10_softlabel.pt',
                            'ImageNet64_imagebatch_1_ipc10_softlabel.pt',
                            'ImageNet64_imagebatch_2_ipc10_softlabel.pt',
                            'ImageNet64_imagebatch_3_ipc10_softlabel.pt',
                            'ImageNet64_imagebatch_4_ipc10_softlabel.pt',
                            ]
        

    ''' load distilled checkpoints'''
    label_syn_mean = []
    for i in range(no_batches):
        distill_data = torch.load(f'checkpoints/{checkpoint_names[i]}')
        decoders[i].load_state_dict(distill_data['vae_state_dict'])
        image_syn[i] = distill_data['syn_img'].clone().detach().requires_grad_(False).to(args.device)
        
        distill_label = torch.load(f'checkpoints/{softlabel_names[i]}')
        label_syn_mean.append(distill_label)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    
    ''' Evaluate '''
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
        if args.dsa:
            print('DSA augmentation strategy: \n', args.dsa_strategy)

        accs_test = []
        accs_train = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

            with torch.no_grad():
                if args.soft_label:
                    image_syn_eval = []
                    label_syn_eval = []
                    label_syn_fixed_eval = []
                    for i in range(no_batches):
                        image_syn_eval.append(image_syn[i].detach())
                        label_syn_eval_batch = F.one_hot(label_syn[i], num_classes).float().detach()
                        label_syn_eval_fixed_batch = label_syn_mean[i].detach()
                        label_syn_eval.append(torch.concat([label_syn_eval_fixed_batch.unsqueeze(1), label_syn_eval_batch.unsqueeze(1)], dim=1))
                        
                else:
                    image_syn_eval = []
                    label_syn_eval = []
                    for i in range(no_batches):
                        image_syn_eval.append(image_syn[i].detach())
                        label_syn_eval.append(label_syn[i].detach())

            args.lr_net = nn.ReLU()(torch.tensor(syn_lr.item()))+ 1.e-4
            _, acc_train, acc_test = evaluate_synset_batch(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
            accs_test.append(acc_test)
            accs_train.append(acc_train)
            del net_eval, image_syn_eval, label_syn_eval
            if args.soft_label:
                del label_syn_fixed_eval
            torch.cuda.empty_cache()

        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std
            save_this_it = True
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=0)
        wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=0)
        wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=0)
        wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='ImageNet64', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--lpc', type=int, default=1, help='laten prior(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_aug', type=float, default=1e-02, help='learning rate for updating augmentation')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--num_subtasks', type=int, default=5, help='number of subtasks for federated distillation')
    
    parser.add_argument('--decoder_size', type=str, default='small', choices=["small", "medium", "large"],
                        help='decoder size, see decoder.py for more info')
    parser.add_argument('--z_dim', type=int, default=64, help='hidden dimension for prototypes')
    parser.add_argument('--kernel_num', type=int, default=128, help='latent dimension for prototypes')
    parser.add_argument('--linear_decode', action=argparse.BooleanOptionalAction, default=True, help='use a linear layer to decode')


    parser.add_argument('--mmd_weight', type=float, default=1.0, help='mmd penalty weight')
    parser.add_argument('--mmd_samples', type=int, default=30, help='number of samples per class')
    parser.add_argument('--load_checkpoint', type=str, default=None, help="load pretrained VAE")
    parser.add_argument('--checkpoint_it', type=int, default=0, help="checkpoint iteration")

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--forward_fixed', action='store_true', help="VAE fixed generation (discrete case)")
    parser.add_argument('--soft_label', action='store_true', help="use soft labels for distillation")

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=10, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--min_start_epoch', type=int, default=0, help='min epoch we can start at')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50 lpc')
    parser.add_argument('--expt_name', type=str, default=None, help='final model')

    parser.add_argument('--run_name', type=str, default=None, help='wandb expt name')
    parser.add_argument('--notes', type=str, default="No description", help='wandb additional notes')

    args = parser.parse_args()

    main(args)


