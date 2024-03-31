# The number of synthesis steps in the current synthesis script is determined by T.
export CUDA_VISIBLE_DEVICES=0
for ((i=300000; i<=300000; i=i+100000))
do
    python3 synthesize.py --source ./preprocessed_data/LJSpeech/val.txt \
                          --restore_step $i \
                          --T 1 \
                          --mode batch --dataset LJSpeech \
                          --model_path ./output/pretrained_model/LJSpeech \
                          --result_path your_output_path/LJ_cm-T1
done