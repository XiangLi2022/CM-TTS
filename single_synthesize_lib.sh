export CUDA_VISIBLE_DEVICES=2
python3 synthesize.py --text "22222222 hello 22222222" \
                      --model naive \
                      --T 1\
                      --speaker_id 3879\
                      --restore_step 300000 \
                      --mode single --dataset LibriTTS \
                      --model_path output/pretrained_model/LibriTTS \
                      --result_path output/single/LibriTTS_cm_T1