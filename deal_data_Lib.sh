export CUDA_VISIBLE_DEVICES=0
python3 prepare_align.py --dataset LibriTTS
nohup python3 preprocess.py --dataset LibriTTS > lib_deal.log