export CUDA_VISIBLE_DEVICES=3
nohup python3 train_cm.py --model consistency_training  \
                          --dataset VCTK \
                          >/your_nohup_log_path/VCTK_cm.log