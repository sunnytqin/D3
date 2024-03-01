# ImageNet with federated distillation
- Small decoder, 1 LPC, 5 subtasks, 1 IPC equivalent:
    - distill: 
    ```python distill_distribution.py --dataset=ImageNet64 --subset=imagebatch_0 --decoder_size=small --lpc=1 --syn_steps=10 --expert_epochs=2 --max_start_epoch=27 --kernel_num=256 --lr_latent=3e-04 --lr_lr=1e-06 --lr_teacher=1e-02 --lr_decode=3e-04 --Iteration 10000 --buffer_path=/n/holyscratch01/barak_lab/Lab/sqin/invariance/results_100_S --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S --epoch_eval_train 2000 --eval_it 200 --load_all --expt_name=mtt_mmd_k256 ``` 
    # need to distill 5 subsets: [imagebatch_0, imagebatch_1, imagebatch_2, imagebatch_3, imagebatch_4]
    - attach softlabels:
    ```python soft_label.py --dataset=ImageNet64  --decoder_size=small --lpc=1 --num_subtasks=5 --expert_epochs=9  --z_dim=64 --kernel_num=256 --lr_teacher=5.e-02 --buffer_path=/n/holyscratch01/barak_lab/Lab/sqin/invariance/results_100_S   --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S --epoch_eval_train 2000 --soft_label```
    - evaluation: 11.5%, 9.7%
    ```python eval_federated.py --dataset=ImageNet64  --decoder_size=small --lpc=1 --num_subtasks=5 --z_dim=64 --kernel_num=256 --lr_teacher=5.e-02 --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S  --epoch_eval_train 2000 --soft_label```
    
- Medium decoder, 2 LPC, 2 subtasks, 2 IPC equivalent:
    - distill: 
    ```python distill_distribution.py --dataset=ImageNet64 --decoder_size=medium --subset=imagenethalf_0 --lpc=2 --syn_steps=10 --expert_epochs=2 --max_start_epoch=10 --z_dim=2048 --lr_latent=5e-03 --lr_lr=5e-06 --lr_teacher=1.5e-02 --lr_decode=8e-05 --Iteration 10000 --buffer_path=/n/holyscratch01/barak_lab/Lab/sqin/invariance/results_100_S --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S --epoch_eval_train 1000  --eval_it=300 --mmd_samples=12 --batch_syn=250 --load_all ```
    # need to distill 2 subsets: [imagenethalf_0, imagenethalf_1]
    - attach softlabels:  
     ```python soft_label.py --dataset=ImageNet64  --decoder_size=medium --lpc=2 --num_subtasks=2 --expert_epochs=12  --z_dim=2048 --lr_teacher=1.4e-02 --buffer_path=/n/holyscratch01/barak_lab/Lab/sqin/invariance/results_100_S  --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S  --epoch_eval_train=2000 --eval_it=1 --soft_label```
    - evaluation: 17.4% ,16.03%
    ```python eval_federated.py --dataset=ImageNet64  --decoder_size=medium --lpc=2 --num_subtasks=2 --z_dim=2048 --lr_teacher=2.e-02  --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S  --epoch_eval_train=2000 --eval_it=1 --soft_label```
    # resnet: (change to smaller LR) `--eval_mode=R, --lr_teacher=1.4e-03`

- Large decoder, 10 LPC, 5 subtasks, 10 IPC equivalent:
    - distill:
    ``` python distill_distribution.py --dataset=ImageNet64 --subset=imagebatch_4 --decoder_size=large --no-linear_decode  --lpc=10 --syn_steps=10 --expert_epochs=2 --max_start_epoch=27 --z_dim=4096 --lr_latent=5e-03 --lr_lr=1e-06 --lr_teacher=8e-03 --lr_decode=8e-05 --Iteration 8000 --buffer_path=/n/holyscratch01/barak_lab/Lab/sqin/invariance/results_100_S --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S --epoch_eval_train 1000 --eval_it 200 --load_all --batch_syn=250 --mmd_samples=20 ```
    # need to distill 5 subsets: [imagebatch_0, imagebatch_1, imagebatch_2, imagebatch_3, imagebatch_4]
    - attach softlabels: 20.3% 18.5(or 9)%
        ```python soft_label.py --dataset=ImageNet64 --decoder_size=large --lpc=10 --num_subtasks=5 --expert_epochs=13 --z_dim=4096   --lr_teacher=5.e-03 --buffer_path=/n/holyscratch01/barak_lab/Lab/sqin/invariance/results_100_S --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --num_eval=1 --eval_mode=S --epoch_eval_train 2000 --soft_label --no-linear_decode```
    - evaluation:
        ```python eval_federated.py --dataset=ImageNet64 --decoder_size=large --lpc=10 --num_subtasks=5 --z_dim=4096 --no-linear_decode --data_path=/n/holyscratch01/barak_lab/Lab/sqin/data/ImageNet/64 --lr_teacher=5.e-03 --num_eval=1 --eval_mode=S --epoch_eval_train 2000 --soft_label ```
        # resnet: (change fewer epochs) `--eval_mode=R, --epoch_eval_train=1000`