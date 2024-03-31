from all_metrics import CalOneModel

if __name__ == "__main__":

    eval_dir_list = list([
        "your/wav_dir",
    ])
    for eval_dir in eval_dir_list:
        cal_one_model_tool = CalOneModel(
            eval_dir,data_type="LJSpeech",
            file_find_type="key_step",
            raw_folder="your/raw_data/LJSpeech/LJSpeech")
        # cal_one_model_tool.get_model_metrics_by_list(["wer"])
        # cal_one_model_tool.get_model_metrics_by_list(["wer_un_comma"])
        
        # cal_one_model_tool.get_model_metrics_by_list(["fid_align_mel", "fid_align_mfcc_un_norm"])

        # cal_one_model_tool.get_model_metrics_by_list(["mcd"])
        # cal_one_model_tool.get_model_metrics_by_list(["speaker_cos"])
        # cal_one_model_tool.get_model_metrics_by_list(["f0_rmse"])
        # cal_one_model_tool.get_model_metrics_by_list(["recall_mfcc"])
        cal_one_model_tool.get_model_metrics_by_list(["ffe", "ssim","mfcc_cos"])