import numpy as np

import audio as Audio
import librosa
import os.path as osp
import os


def find_key_wav_files(path_dir):
    wav_files = []
    for train_step in [100000, 200000, 300000]:
        wav_files.extend(find_wav_files(osp.join(path_dir, str(train_step))))
    return wav_files


def find_wav_files(path_dir):
    wav_files = []
    for root, dirs, files in os.walk(path_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files


def save_mel_cache(wav_filepath):
    """
    这里并没有缓存逻辑
    :param wav_filepath:
    :return:
    """

    def get_cache_filepath():
        wav_dir = osp.dirname(wav_filepath)
        cache_dir = osp.join(wav_dir + "_mel")
        base_name = osp.basename(wav_filepath) + ".npy"
        os.makedirs(cache_dir, exist_ok=True)
        return osp.join(cache_dir, base_name)

    STFT = Audio.stft.TacotronSTFT(
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0,
        mel_fmax=8000,
    )
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(librosa.load(wav_filepath)[0], STFT)
    mel_spectrogram = mel_spectrogram.T
    np.save(get_cache_filepath(), mel_spectrogram)


def get_mel(deal_dir_list, find_type=None):
    if find_type is not None and find_type == "key_step":
        find_fun = find_key_wav_files
    else:
        find_fun = find_wav_files
    for deal_dir in deal_dir_list:
        for wav_filepath in find_fun(path_dir=deal_dir):
            save_mel_cache(wav_filepath)


if __name__ == '__main__':
    deal_dir_list = list([
        "/home/lixiang/lx/output/result/VCTK_cm_1110_10_l2_loss-second-moment_cm",
        "/home/lixiang/lx/output/result/VCTK_cm_1111_10_melloss_loss-second-moment_cm"
    ])
    get_mel(deal_dir_list, find_type="key_step")
