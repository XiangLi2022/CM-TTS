import librosa
import numpy as np
import scipy
import torch
import yaml

from .mb_model import MBNet
from .ld_model.LDNet import LDNet


class MOSCal:
    def __init__(self, sample_rate=22500):
        self.sample_rate = sample_rate
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # 使用显卡
        else:
            self.device = torch.device("cpu")

        self.ld_net_model = None
        self.mb_net_model = None

    def __load_mb_model(self):
        mb_model_path = "/Users/bufan/PycharmProjects/Pytorch-MBNet-main/pre_trained/model-50000.pt"

        mb_net_model = MBNet(num_judges=5000).to(self.device)
        mb_net_model.load_state_dict(torch.load(mb_model_path, map_location="cpu"))
        return mb_net_model

    def __load_ld_model(self):
        ld_model_path = "/Users/bufan/PycharmProjects/LDNet-main/exp/Pretrained-LDNet-ML-2337/model-27000.pt"
        ld_config_path = "/Users/bufan/PycharmProjects/LDNet-main/exp/Pretrained-LDNet-ML-2337/config.yml"
        with open(ld_config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        ld_net_model = LDNet(config).to(self.device)
        ld_net_model.load_state_dict(torch.load(ld_model_path, map_location="cpu"), strict=False)
        ld_net_model.eval()
        return ld_net_model

    def get_ld_mos(self, wav_path):
        if self.ld_net_model is None:
            self.ld_net_model = self.__load_ld_model()
        wav, _ = librosa.load(wav_path, sr=self.sample_rate, )
        wav = torch.tensor(
            np.abs(librosa.stft(wav, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)).T)
        wav = wav.to(self.device).unsqueeze(0)

        pred_mean_scores, posterior_scores = self.ld_net_model.average_inference(
            spectrum=wav,
            include_meanspk=False
        )
        return pred_mean_scores.detach().numpy()[0]

    def get_mb_mos(self, wav_path):
        if self.mb_net_model is None:
            self.mb_net_model = self.__load_mb_model()
        wav, _ = librosa.load(wav_path, sr=self.sample_rate, )
        wav = torch.tensor(
            np.abs(librosa.stft(wav, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)).T)
        wav = wav.to(self.device)
        # print(wav.size())
        wav = wav.unsqueeze(0).unsqueeze(1)
        mean_scores = self.mb_net_model.get_mean_mos(
            spectrum=wav,
        )
        # 他是每个帧预测一个，所以求一个均值
        return torch.mean(mean_scores).detach().numpy()


if __name__ == "__main__":
    wav_path = "/Users/bufan/PycharmProjects/Pytorch-MBNet-main/vcc2018_evaluation/VCC2SF3/30002.wav"
    mos_tool = MOSCal()
    mos_mean = mos_tool.get_ld_mos(wav_path)
    print(mos_mean)
