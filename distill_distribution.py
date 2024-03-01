import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
import wandb
import copy
import random

from reparam_module import ReparamModule
from mmd import mix_rbf_mmd2_fixed
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, get_memory, count_parameters, DiffAugment, ParamDiffAug
from decoder import TinyVAE, VanillaVAE

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = (np.arange(0, args.Iteration + 1, args.eval_it) + args.checkpoint_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if args.dsa:
        args.dc_aug_param = None

    ''' initialize the decoder'''
    if args.decoder_size == 'small': # for all image sizes
        decoder = TinyVAE("test", im_size[0], channel, kernel_num=args.kernel_num, z_size=args.z_dim)
    elif args.decoder_size == 'medium':
        if args.dataset in ["CIFAR100", "CIFAR10"]: # smaller image size
            decoder = VanillaVAE(channel, latent_dim=args.z_dim, linear_decode=args.linear_decode, hidden_dims=[64, 128, 256, 512])
        else: 
            decoder = VanillaVAE(channel, latent_dim=args.z_dim, linear_decode=args.linear_decode)
    elif args.decoder_size == 'large': # only for 64X64 (TinyImageNet and ImageNet)
        decoder = VanillaVAE(channel, latent_dim=args.z_dim, linear_decode=args.linear_decode, hidden_dims=[64, 128, 256, 512, 1024])

    decoder.to(args.device)
    args.decoder = decoder

    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(
        sync_tensorboard=False,
        project="project-name",  
        entity="entity", 
        config=args,
        name=args.run_name,
        notes=args.notes,
        mode="disabled",
        )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.decoder = decoder
    args.zca_trans = zca_trans
    # args.diffaug_param = diffaug_param

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.lpc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Decoder: ', count_parameters(args.decoder))
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    if args.mmd_weight > 1.0e-4:
        if args.dataset != "ImageNet64":
            images_all = []
            labels_all = []
            indices_class = [[] for c in range(num_classes)]
            print("BUILDING DATASET")
            for i in tqdm(range(len(dst_train))):
                sample = dst_train[i]
                images_all.append(torch.unsqueeze(sample[0], dim=0))
                labels_all.append(class_map[torch.tensor(sample[1]).item()])
            for i, lab in tqdm(enumerate(labels_all)):
                indices_class[lab].append(i)
        
            images_all = torch.cat(images_all, dim=0).to("cpu")
            labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
        elif args.dataset == "ImageNet64":
            images_all = dst_train.images.detach()
            labels_all = dst_train.labels.detach()
            indices_class = [[] for c in range(num_classes)]

            for i, lab in tqdm(enumerate(labels_all)):
                indices_class[lab].append(i)
        
            images_all = images_all.to("cpu")
            labels_all = labels_all.to("cpu")
        else: 
            raise AssertionError("Invalid dataset")

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    label_syn = torch.tensor(np.array([np.ones(args.lpc,dtype=np.int_)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    latent_syn = torch.randn(size=(num_classes * args.lpc, args.z_dim, 2))

    ''' load checkpoint for continuting training'''
    if args.load_checkpoint:
        distill_data = torch.load(f'checkpoints/{args.load_checkpoint}')
        decoder.load_state_dict(distill_data['vae_state_dict'])
        if latent_syn.shape[0] == distill_data['syn_img'].shape[0]:
            latent_syn = distill_data['syn_img']
        else:
            latent_syn[0:distill_data['syn_img'].shape[0], :, :] = distill_data['syn_img']

    ''' initialize optimizers'''
    latent_syn = latent_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    
    optimizer_latent = torch.optim.SGD([latent_syn], lr=args.lr_latent, momentum=0.5)
    optimizer_latent.zero_grad()
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_lr.zero_grad()
    optimizer_decode = torch.optim.Adam(decoder.parameters(), lr=args.lr_decode, weight_decay=1e-05)
    optimizer_decode.zero_grad()
    

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    ''' define expert trajectories folder '''
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet" or args.dataset == "ImageNet64":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset == "Tiny":
        expert_dir = os.path.join(expert_dir, args.subset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    ''' load expert trajectories if enough CPU memory'''
    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}


    ''' main distillation loop'''
    for it in range(args.checkpoint_it, args.checkpoint_it+args.Iteration+1):
        save_this_it = False

        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = latent_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    # args.lr_net = syn_lr.item()
                    args.lr_net = nn.ReLU()(torch.tensor(syn_lr.item()))+ 1.e-4
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                    del net_eval, image_syn_eval, label_syn_eval
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
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        ''' evaluate distilled data'''
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = latent_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(latent_syn.detach().cpu()))}, step=it)

                if args.lpc < 50 or args.force_save:
                    # save mean for latent prior
                    upsampled = decoder.forward_fixed(image_save)
                    augmented = [upsampled[None, :]]
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)
                    
                    # save random samples 
                    if (args.lpc < 10 and num_classes < 200):
                        for _ in range(9):
                            augmented.append(decoder(image_save)[None, :])

                        augmented = torch.vstack(augmented).transpose(0, 1).flatten(0, 1)
                        grid = torchvision.utils.make_grid(augmented, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Augmented_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca: # used only for CIFAR-10
                        image_save = decoder.forward_fixed(image_save).to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)
                            

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                # print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)
        
        grand_loss = 0

        ''' Dual Training Objective 1/2 - MTT '''
        if args.mtt_weight > 1e-3:
            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_trajectory[start_epoch]

            target_params = expert_trajectory[start_epoch+args.expert_epochs]
            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

            student_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)

            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

            syn_latents = latent_syn

            y_hat = label_syn.to(args.device)

            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for step in range(args.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(syn_latents))
                    indices_chunks = list(torch.split(indices, args.batch_syn))

                these_indices = indices_chunks.pop()

                x = syn_latents[these_indices]
                this_y = y_hat[these_indices]

                if args.dsa and (not args.no_aug):
                    if args.forward_fixed:
                        x = decoder.forward_fixed(x)
                    else:
                        x = decoder(x)

                if args.distributed:
                    forward_params = student_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params

                x = student_net(x, flat_param=forward_params)
                ce_loss = criterion(x, this_y)

                grad = torch.autograd.grad(ce_loss, student_params, create_graph=True)[0]

                lr = nn.LeakyReLU()(syn_lr) + 1.e-4
                student_params = student_params - lr * grad

            param_loss = torch.tensor(0.0).to(args.device)
            param_dist = torch.tensor(0.0).to(args.device)

            param_loss += torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            mtt_loss = param_loss

            grand_loss += mtt_loss.detach().cpu()
            
            optimizer_latent.zero_grad()
            optimizer_lr.zero_grad()
            optimizer_decode.zero_grad()

            mtt_loss.backward()

            optimizer_latent.step()
            optimizer_lr.step()
            optimizer_decode.step()
            wandb.log({"MTT_Loss": mtt_loss.detach().cpu()}, step=it)
            
        
        ''' Dual Training Objective 2/2 - MMD '''
        if args.mmd_weight > 1e-4: 
            syn_latents = latent_syn
            feature_only = 2 # feature level 
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
            if args.mmd_param == 'final': # does not work as well
                final_params = expert_trajectory[-1]
            elif args.mmd_param == 'expert':
                final_params = expert_trajectory[args.max_start_epoch + args.expert_epochs]
            else:
                raise AssertionError("Invalid mmd_param")

            final_params = torch.cat([p.data.to(args.device).reshape(-1) for p in final_params], 0)
            
            if args.distributed:
                final_params = final_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                final_params = final_params

            num_samples = args.mmd_samples // args.lpc  # reduce number of samples avoids OOM error
            if args.forward_fixed:
                synthetic_samples = decoder.forward_fixed(syn_latents).unsqueeze(1)
            else:
                synthetic_samples = []
                for _ in range(num_samples):
                    synthetic_samples.append(decoder(syn_latents).unsqueeze(1))
                synthetic_samples = torch.concatenate(synthetic_samples, dim=1)

            mmd_loss = 0
            mmd_loss_total = 0
            for c in range(num_classes):
                real_images = get_images(c, num_samples * args.lpc).to(syn_latents.device)
                activation_real = student_net(real_images, flat_param=final_params, feature_only=feature_only).detach()
                activation_real.requires_grad = False
            
                # make a copy 
                synthetic_samples_class = synthetic_samples[c*args.lpc: (c+1)*args.lpc , :, :, :, :].to(syn_latents.device)
                activation_syn = student_net(synthetic_samples_class.reshape(-1,  *synthetic_samples.shape[2:]), 
                                             flat_param=final_params, feature_only=feature_only)

                # RBF kernel loss
                mmd_loss += mix_rbf_mmd2_fixed(activation_syn, activation_real)

            mmd_loss = args.mmd_weight * mmd_loss
    
            optimizer_latent.zero_grad()
            optimizer_decode.zero_grad()

            mmd_loss.backward()

            optimizer_latent.step()
            optimizer_decode.step()
            grand_loss += mmd_loss.detach().cpu()
            mmd_loss_total += mmd_loss.detach().cpu()

            mmd_loss = 0.
            wandb.log({"MMD_Loss": mmd_loss_total.detach().cpu()}, step=it)

        wandb.log({"Grand_Loss": grand_loss.detach().cpu()}, step=it)

        if args.mtt_weight > 0: 
            for _ in student_params:
                del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
        
        # assert syn_lr > 0.

        ''' save model checkpoints '''
        if args.expt_name is not None and it % 2000 == 0 and it > 0:
            save_path = os.path.join('checkpoints', f'{args.dataset}_{args.subset}_lpc{args.lpc}_{args.expt_name}_step{it}.pt')
        
            torch.save({
                    'syn_img': latent_syn,
                    'vae_state_dict': decoder.state_dict(),
                    'z_dim': args.z_dim,
                    'kernel_num': args.kernel_num, 
                    'im_size': im_size, 
                    'channel': channel, 
                    }, save_path)
    

    wandb.finish()

    ''' save final mode checkpoints '''
    if args.expt_name is not None:
        save_path = os.path.join('checkpoints', f'{args.dataset}_{args.subset}_lpc{args.lpc}_{args.expt_name}_final.pt')

        torch.save({
                'syn_img': latent_syn,
                'vae_state_dict': decoder.state_dict(),
                'z_dim': args.z_dim,
                'kernel_num': args.kernel_num, 
                'im_size': im_size, 
                'channel': channel, 
                }, save_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--lpc', type=int, default=1, help='latent prior(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_latent', type=float, default=1000, help='learning rate for updating synthetic latent priors')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_decode', type=float, default=1e-02, help='learning rate for updating... decoder')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--decoder_size', type=str, default='small', choices=["small", "medium", "large"],
                        help='decoder size, see decoder.py for more info')
    parser.add_argument('--z_dim', type=int, default=64, help='hidden dimension for latent priors')
    parser.add_argument('--kernel_num', type=int, default=128, help='latent dimension for latent priors, small decoder param')
    parser.add_argument('--linear_decode', action=argparse.BooleanOptionalAction, default=True, help='use a linear layer to decode, large/medium decoder param')
    
    parser.add_argument('--mtt_weight', type=float, default=1.0, help='use mtt')
    parser.add_argument('--mmd_weight', type=float, default=1.0, help='mmd penalty weight')
    parser.add_argument('--mmd_param', type=str, default='final', choices=['final', 'expert'], help='which expert tagging to use for mmd')
    parser.add_argument('--mmd_samples', type=int, default=30, help='number of samples per class for MMD computation')
    
    
    parser.add_argument('--load_checkpoint', type=str, default=None, help="load pretrained decoder")
    parser.add_argument('--checkpoint_it', type=int, default=0, help="checkpoint iteration")

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--forward_fixed', action='store_true', help="decoder fixed generation (distilling only the mean)")
    parser.add_argument('--soft_label', action='store_true', help="not used here")

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--min_start_epoch', type=int, default=0, help='min epoch we can start at')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50 lpc')
    parser.add_argument('--expt_name', type=str, default=None, help='pass in name to save model checkpoints')

    parser.add_argument('--run_name', type=str, default=None, help='wandb expt name')
    parser.add_argument('--notes', type=str, default="No description", help='wandb additional notes')

    args = parser.parse_args()

    main(args)


