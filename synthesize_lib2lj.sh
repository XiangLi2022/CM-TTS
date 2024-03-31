
export CUDA_VISIBLE_DEVICES=10
for ((i=300000; i<=300000; i=i+10000))
do
    nohup python3 synthesize_zeroshot_lj.py --source your/preprocessed_data/LJSpeech/val.txt \
                          --model naive \
                          --T 2\
                          --restore_step $i \
                          --mode batch --dataset LibriTTS \
                          --model_path your/ckpt/Lib_cm \
                          --result_path your/zero_shot_testLJ/Lib_cm$i
done