
export CUDA_VISIBLE_DEVICES=0
for ((i=300000; i<=300000; i=i+100000))
do
    nohup python3 synthesize_zeroshot_vctk.py --source your/preprocessed_data/VCTK/val.txt \
                          --model naive \
                          --T 1\
                          --restore_step $i \
                          --mode batch --dataset LibriTTS \
                          --model_path your/ckpt/LJ_cm \
                          --result_path your/zero_shot/Lib_cm
done