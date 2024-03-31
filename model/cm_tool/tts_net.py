# 直接用原有的扩散模型核心生成一个套皮的CM
import torch.nn as nn
import torch as th

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from ..modules import Denoiser
from ..diffgantts import DurationPitchSpeakerNet
from ..loss import CMLoss


class CMDenoiserTTS(nn.Module):
    def __init__(self, use_fp16, **kwargs):
        super().__init__()
        self.net = Denoiser(**kwargs)
        self.dtype = th.float16 if use_fp16 else th.float32

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.net.apply(convert_module_to_f16)  # 就是循环给所有的模型都用这个东西

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.net.apply(convert_module_to_f32)

    def forward(self, x, timesteps, conditioner=None,speaker_emb=None, mask=None):
        # 主要是对齐一下数据接口
        x = x.transpose(2, 3)
        conditioner =conditioner.transpose(1, 2)
        net_out = self.net(mel=x,  diffusion_step=timesteps, 
        conditioner=conditioner,speaker_emb=speaker_emb, mask=mask)
        # print(net_out.size())
        net_out = net_out.transpose(2, 3)
        return net_out


class CMTotalTTS(nn.Module):
    def __init__(self, use_fp16, args, preprocess_config, model_config, train_config, **kwargs):
        super().__init__(**kwargs)
        self.duration_pitch_energy_net = DurationPitchSpeakerNet(args, preprocess_config, model_config, train_config)
        self.loss_cal_tool = CMLoss(args, preprocess_config, model_config, train_config)
        self.net = Denoiser(preprocess_config, model_config)
        self.dtype = th.float16 if use_fp16 else th.float32
        self.losses = None

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.duration_pitch_energy_net.apply(convert_module_to_f16)
        self.net.apply(convert_module_to_f16)  # 就是循环给所有的模型都用这个东西

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.duration_pitch_energy_net.apply(convert_module_to_f32)
        self.net.apply(convert_module_to_f32)

    def get_tts_loss(self):
        return self.losses

    def get_segmentation_model(self):
        """"""
        def denoise_fun(mel, diffusion_step, conditioner, speaker_emb, mask=None):
            mel = mel.transpose(2, 3)
            conditioner = conditioner.transpose(1, 2)
            net_out = self.net(mel, diffusion_step, conditioner, speaker_emb, mask=mask)
            return net_out.transpose(2, 3)
        return self.duration_pitch_energy_net, denoise_fun

    def forward(self, x, timesteps,
                speakers,  # 2
                texts,  # 3
                src_lens,  # 4
                pitch=None,
                f0=None,
                uv=None,
                cwt_spec=None,
                f0_mean=None,
                f0_std=None,
                # max_src_len,  # 5
                mel_lens=None,  # 6
                # p_targets=None,  # 8
                e_targets=None,  # 9
                d_targets=None,  # 10
                mel2phs=None,  # 11
                spker_embeds=None,  # 为了和diffgan融合增加的
                p_control=1.0,
                e_control=1.0,
                d_control=1.0,
                # conditioner=None,speaker_emb=None, mask=None
                **kwargs
                ):
        """

        :param x: torch.Size([batch, 1, seq_len, 80])  这个就是直接从batch读出来那个
        :param timesteps:
        :param speakers:
        :param texts:
        :param src_lens:
        :param max_src_len:
        :param mel_lens:
        :param max_mel_len:
        :param p_targets:
        :param e_targets:
        :param d_targets:
        :param mel2phs:
        :param spker_embeds:
        :param p_control:
        :param e_control:
        :param d_control:
        :return:torch.Size([batch, 1, seq_len, 80])
        """
        # 数据对齐操作
        if pitch is None:
            p_targets = None
        else:
            p_targets = dict({
            "pitch": pitch,
            "f0": f0,
            "uv": uv,
            "cwt_spec": cwt_spec,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
        })
        # print(pitch)
        # 条件信息提取网络
        out_dict = self.duration_pitch_energy_net(
            speakers=speakers,  # 2
            texts=texts,  # 3
            src_lens=src_lens,  # 4
            # max_src_len=max_src_len,  # 5
            mels=x,  # 6  (batch_size,max_seq_len,80)
            mel_lens=mel_lens,
            p_targets=p_targets,
            e_targets=e_targets,
            d_targets=d_targets,
            mel2phs=mel2phs,
            spker_embeds=spker_embeds,  # 为了和diffgan融合增加的
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )
        conditioner = out_dict["cond"]  # torch.Size([batch_size, seq_len, 256])
        speaker_emb = out_dict["speaker_emb"]
        mask = out_dict["mel_masks"]
        # 条件信息控制结果预测
        x = x.transpose(2, 3)  # torch.Size([batch_size, 1, 80, seq_len])
        conditioner = conditioner.transpose(1, 2)
        net_out = self.net(
            mel=x,  diffusion_step=timesteps,
            conditioner=conditioner, speaker_emb=speaker_emb, mask=mask)
        net_out = net_out.transpose(2, 3)  # torch.Size([batch_size, 1, seq_len, 80])
        # net_out = net_out[:, 0]

        if p_targets is not None:
            # 计算TTS的loss
            loss_label_list = [None]*14
            loss_label_list[3] = texts
            loss_label_list[6] = x
            loss_label_list[9] = p_targets  # 9
            loss_label_list[10] = e_targets  # 10
            loss_label_list[11] = d_targets  # 11
            loss_label_list[12] = mel2phs  # 12
            mel_predictions = None
            loss_calculate_dict = {
                "pitch_predictions": out_dict["p_predictions"],
                "energy_predictions": out_dict["e_predictions"],
                "log_duration_predictions": out_dict["log_d_predictions"],
                "src_masks": out_dict["src_masks"],
                "mel_masks": out_dict["mel_masks"],
                "mel_predictions": mel_predictions,
            }
            self.losses = self.loss_cal_tool(loss_label_list, loss_calculate_dict)
        

        # 计算额外的loss

        return net_out
