export CUDA_VISIBLE_DEVICES=0
python3 prepare_align.py --dataset VCTK
nohup python3 preprocess.py --dataset VCTK > vctk_deal.log