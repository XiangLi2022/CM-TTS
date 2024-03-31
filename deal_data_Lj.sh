export CUDA_VISIBLE_DEVICES=0
python3 prepare_align.py --dataset LJSpeech
nohup python3 preprocess.py --dataset LJSpeech > lj_deal.log