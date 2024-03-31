export CUDA_VISIBLE_DEVICES=3
python3 p_rtf_cm.py --source your/preprocessed_data/VCTK/val.txt \
                      --model naive \
                      --restore_step 300000 \
                      --mode batch --dataset VCTK \
                      --model_path your/ckpt/VCTK_cm \
                      --result_path your/result/VCTK_cm
                      # --cut true \