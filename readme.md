# Dataset Distillation by Matching Training Trajectories

<!-- ### [Project Page](https://georgecazenavette.github.io/mtt-distillation/) | [Paper](https://arxiv.org/abs/2203.11932) -->
<br>

This repo contains code for training expert trajectories and distilling synthetic data from our Distributional Dataset Distillation with subtask decomposition. 

<img src='docs/D3_demo.gif' width=600>

### Getting Started

First, download our repo:
```bash
git clone 
cd distributional-distillation
```

The following pacakages are needed in your environment:   
> torch==1.13.1
> torchvision==0.14.1
> kornia==0.6.12
> einops==0.6.1
> numpy==1.20.1
> tqdm==4.64.1
> wandb==0.13.8
> scipy==1.10.1
See `requirements.txt` for an exhaustive list of dependencies

### Evaluate our Distilled Distribution
Download our distilled distribution from link, and place under folder `checkpoints`
The following command evaluate the distilled distribution (smallest budget) on ImageNet-1K: 
```bash
```

## Disttributional Distillation Pipeline

### Generating Subset Experts
Before doing any distillation, you'll need to generate some expert trajectories using ```buffer.py``` on each of the subsets
The following command will train 100 ConvNet models on ImageNet subsets: 
```bash
python buffer.py --dataset=ImageNet64 --subset=imagebatch_0 --model=ConvNet --train_epochs=10 --num_experts=100  --buffer_path={custom_buffer_path} --data_path={path_to_ImageNet64_dataset} --save_interval 10
```

### Distributional Distillation on subtasks 
The following command distill each of the subsets:
```bash
python distill_distribution.py --dataset=ImageNet64 --subset=imagebatch_0 --decoder_size=small --lpc=1 --syn_steps=10 --expert_epochs=2 --max_start_epoch=27 --kernel_num=256 --lr_latent=3e-04 --lr_lr=1e-06 --lr_teacher=1e-02 --lr_decode=3e-04 --Iteration 10000 --buffer_path={custom_buffer_path} --data_path={path_to_ImageNet64_dataset} --num_eval=1 --eval_mode=S --epoch_eval_train 2000 --eval_it 200 --load_all --expt_name=small
```

### Assign softlabel and evaluate after all substasks have been distilled
```bash
python soft_label.py --dataset=ImageNet64  --decoder_size=small --lpc=1 --num_subtasks=5 --expert_epochs=9  --z_dim=64 --kernel_num=256 --lr_teacher=5.e-02 --buffer_path={custom_buffer_path}  --data_path={path_to_ImageNet64_dataset} --num_eval=1 --eval_mode=S --epoch_eval_train 2000 --soft_label
```

### Full reproducibility
See `clearn_script.md` for commands (and non-default hyperprameters used) to distill and evaluate the distilled distribution on ImageNet and TinyImagNet
