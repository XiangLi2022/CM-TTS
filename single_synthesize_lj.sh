export CUDA_VISIBLE_DEVICES=2
python3 synthesize.py --text "22222222 hello 22222222" \
                      --model naive \
                      --T 1\
                      --restore_step 300000 \
                      --mode single --dataset LJSpeech \
                      --model_path output/pretrained_model/LJSpeech \
                      --result_path output/single/LJ_cm_T1