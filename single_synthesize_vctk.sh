export CUDA_VISIBLE_DEVICES=2
python3 synthesize.py --text "22222222 hello 22222222" \
                      --model naive \
                      --T 1\
                      --speaker_id p282\
                      --restore_step 300000 \
                      --mode single --dataset VCTK \
                      --model_path output/pretrained_model/VCTK \
                      --result_path output/single/VCTK_cm_T1