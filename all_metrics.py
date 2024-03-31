import os
import re
import time

import joblib
import librosa
import re
import numpy as np
import pandas as pd
import torch
import pyworld
import pysptk
import math
from fastdtw import fastdtw
from functools import partial
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse
import sys
from speakerembedder import PreDefinedEmbedder
from metrics.f0_frame_error import FFE
from metrics.mos import MOSCal
from metrics.fid import CalFidSeries, CalRecall, CalPrecision, CalFIDAlign
from utils.tools import pad_1D
import os.path as osp
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import librosa
from scipy.stats import entropy


class Cal:
    def __init__(
            self,
            syn_folder,
            data_type="VCTk",
            raw_folder="your/raw_data/VCTK",
            mel_npy_path="your/result/mel_npy_cache",
            SAMPLING_RATE=22050, FRAME_PERIOD=5.0,
            clear_cache=False,
    ):
        """

        :param syn_folder:
        :param data_type:
        :param raw_folder:
        :param mel_npy_path:
        :param SAMPLING_RATE:
        :param FRAME_PERIOD:
        :param clear_cache:
        """
        self.syn_folder = syn_folder
        os.makedirs(self.syn_folder, exist_ok=True)
        self.raw_folder = raw_folder
        self.mel_npy_path = osp.join(mel_npy_path, str(int(time.time())))
        self.data_type = data_type
        self.syn2label = self.__init_file_match()
        self.SAMPLING_RATE = SAMPLING_RATE
        self.FRAME_PERIOD = FRAME_PERIOD
        self.mos_tool: MOSCal = None
        self.clear_cache = clear_cache
        self.device = "cuda"

    def __read_wav(self, file_path):
        return librosa.load(file_path, sr=self.SAMPLING_RATE, mono=True)[0]

    def __get_mgc(self, file_path):
        alpha = 0.435  # 0.65 commonly used at 22050 Hz  #0.44
        fft_size = 512
        mcep_size = 24
        wav = self.__read_wav(file_path)
        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=self.SAMPLING_RATE,
                                     frame_period=self.FRAME_PERIOD, fft_size=fft_size)
        # Extract MCEP features
        return pysptk.sptk.mcep(
            sp, order=mcep_size, alpha=alpha, maxiter=0,
            etype=1, eps=1.0E-8, min_det=0.0, itype=3
        )

    def __get_f0(self, wav_filepath):
        wav = self.__read_wav(wav_filepath)
        wav = wav.astype(np.float64)
        f0, _ = pyworld.harvest(wav, self.SAMPLING_RATE, frame_period=self.FRAME_PERIOD, f0_floor=71.0, f0_ceil=800.0)
        return f0

    def __get_align_f0(self, file_pair):
        # 获取f0
        f0_1 = self.__get_f0(file_pair[0])
        f0_2 = self.__get_f0(file_pair[1])

        # 只选取发声的部分
        f0_1 = f0_1[f0_1 > 0].reshape(1, -1)
        f0_2 = f0_2[f0_2 > 0].reshape(1, -1)

        # 进行dtw对齐
        _, path = fastdtw(f0_1.T, f0_2.T)
        aligned_f0_1 = f0_1[:, [p[0] for p in path]].T.reshape(-1)
        aligned_f0_2 = f0_2[:, [p[1] for p in path]].T.reshape(-1)
        return aligned_f0_1, aligned_f0_2

    def __get_mfcc(self, filepath):
        mfcc = librosa.feature.mfcc(y=librosa.load(filepath)[0], sr=self.SAMPLING_RATE).T  # (seq_len,20)

        # 归一化对齐后的MFCC特征
        return mfcc / np.linalg.norm(mfcc, axis=0)  # (seq_len,20)

    @staticmethod
    def __get_gmm_kl(wav_filepath_pair, get_feature_fun):
        """
        这个必须是标签在前，预测值在后。
        参考链接：http://t.csdnimg.cn/QZqQu
        :param get_feature_fun: 获得特征的方法
        :param wav_filepath_pair:
        :return:
        """
        feature_target = get_feature_fun(wav_filepath_pair[0])  # (seq_len,20)
        gmm_target = GaussianMixture(n_components=30, covariance_type='full')
        gmm_target.fit(feature_target)

        feature_pre = get_feature_fun(wav_filepath_pair[0])  # (seq_len,20)
        gmm_pre = GaussianMixture(n_components=30, covariance_type='full')
        gmm_pre.fit(feature_pre)
        kl_ = entropy(gmm_target.score_samples(feature_target), gmm_pre.score_samples(feature_target))
        return 0 if kl_ == np.inf else kl_

    def __init_file_match(self, ):
        match_files = dict()
        target_extension = ".wav"
        syn_names = os.listdir(self.syn_folder)
        filtered_syn_names = [file for file in syn_names if file.endswith(target_extension)]
        filtered_syn_names = [path for path in filtered_syn_names if not path.endswith("_16000.wav")]
        for syn_name in filtered_syn_names:
            syn_raw_file = os.path.join(self.syn_folder, syn_name)
            if self.data_type == "VCTK":
                raw_folder_name = str(syn_name).split("-")[0]
                raw_wav_file = os.path.join(self.raw_folder, raw_folder_name, syn_name)
            elif self.data_type == "LJSpeech":
                raw_wav_file = os.path.join(self.raw_folder, syn_name)
            else:
                raise NotImplementedError
            match_files.update({syn_raw_file: raw_wav_file})
        return match_files

    def compute_mfcc_gmm_kl(self):
        deal_pair_fun = partial(self.__get_gmm_kl, get_feature_fun=self.__get_mfcc)
        return np.mean(np.array(list(map(
            deal_pair_fun,
            zip(
                self.syn2label.values(),
                self.syn2label.keys()))
        )))

    def __get_align_fid_tool(self):
        target_list = list(self.syn2label.values())
        generate_list = list(self.syn2label.keys())
        fid_cal_tool = CalFIDAlign(generate_filepath_list=generate_list, target_filepath_list=target_list)
        return fid_cal_tool

    def compute_fid_align_mfcc(self):
        fid_cal_tool = self.__get_align_fid_tool()
        return fid_cal_tool(norm=True, feature_type="mfcc_un_norm")

    def compute_fid_align_mfcc_un_norm(self):
        fid_cal_tool = self.__get_align_fid_tool()
        return fid_cal_tool(norm=False, feature_type="mfcc_un_norm")

    def compute_fid_align_mel(self):
        fid_cal_tool = self.__get_align_fid_tool()
        return fid_cal_tool(norm=False, feature_type="mel")

    def compute_wer_un_comma(self):
        import whisper
        import jiwer
        model = whisper.load_model("large", device=self.device)

        base_name2text = dict()

        def fill_base_name2text(val_txt_path):
            with open(val_txt_path, 'r') as file:
                for line in file:
                    parts = line.strip().split('|')
                    if len(parts) >= 4:
                        with torch.no_grad():
                            identifier, text = parts[0], parts[3]
                            base_name2text[identifier] = text

        if self.data_type == "VCTK":
            val_txt_path = "/your/preprocessed_data/VCTK/val.txt"
            fill_base_name2text(val_txt_path)
            train_txt_path = "/your/preprocessed_data/VCTK/train.txt"
            fill_base_name2text(train_txt_path)

        elif self.data_type == "LJSpeech":
            val_txt_path = "/your/preprocessed_data/LJSpeech/val.txt"
            fill_base_name2text(val_txt_path)
            train_txt_path = "/your/preprocessed_data/LJSpeech/train.txt"
            fill_base_name2text(train_txt_path)
        else:
            raise NotImplementedError

        def get_text_from_wav_path(wav_path):
            base_name = osp.basename(wav_path).split(".")[0]
            syn_text = model.transcribe(wav_path)["text"][1:]
            syn_text = syn_text.lower()
            return base_name2text[base_name].replace(",", ""), syn_text.replace(",", "")

        ground_truth = []
        hypothesis = []
        for wav_file_path in self.syn2label.keys():
            label_text, syn_text = get_text_from_wav_path(wav_file_path)
            ground_truth.append(label_text)
            hypothesis.append(syn_text)
        file = open(osp.join(osp.dirname(list(self.syn2label.keys())[0]), "Awer_output_un_comma.txt"), "w")
        for label_string, syn_string in zip(ground_truth, hypothesis):
            file.write("真值是" + label_string + "\n")
            file.write("预测是" + syn_string + "\n")
        file.close()
        wer = jiwer.wer(
            ground_truth,
            hypothesis)

        return wer


    def compute_wer(self):
        import whisper
        import jiwer
        model = whisper.load_model("large", device=self.device)

        base_name2text = dict()

        def fill_base_name2text(val_txt_path):
            with open(val_txt_path, 'r') as file:
                for line in file:
                    parts = line.strip().split('|')
                    if len(parts) >= 4:
                        with torch.no_grad():
                            identifier, text = parts[0], parts[3]
                            base_name2text[identifier] = text

        if self.data_type == "VCTK":
            val_txt_path = "/your/preprocessed_data/VCTK/val.txt"
            fill_base_name2text(val_txt_path)
            train_txt_path = "/your/preprocessed_data/VCTK/train.txt"
            fill_base_name2text(train_txt_path)

        elif self.data_type == "LJSpeech":
            val_txt_path = "/your/preprocessed_data/LJSpeech/val.txt"
            fill_base_name2text(val_txt_path)
            train_txt_path = "/your/preprocessed_data/LJSpeech/train.txt"
            fill_base_name2text(train_txt_path)
        else:
            raise NotImplementedError

        def get_text_from_wav_path(wav_path):
            base_name = osp.basename(wav_path).split(".")[0]
            syn_text = model.transcribe(wav_path)["text"][1:]
            syn_text = syn_text.lower()
            return base_name2text[base_name], syn_text

        ground_truth = []
        hypothesis = []
        for wav_file_path in self.syn2label.keys():
            label_text, syn_text = get_text_from_wav_path(wav_file_path)
            ground_truth.append(label_text)
            hypothesis.append(syn_text)
        file = open(osp.join(osp.dirname(list(self.syn2label.keys())[0]), "Awer_output.txt"), "w")
        for label_string, syn_string in zip(ground_truth, hypothesis):
            file.write("真值是" + label_string + "\n")
            file.write("预测是" + syn_string + "\n")
        file.close()
        wer = jiwer.wer(
            ground_truth,
            hypothesis)

        return wer

    def compute_si_sdr(self):
        def cal_pair(filepath_pair):
            """Scale-Invariant Signal to Distortion Ratio (SI-SDR)
                    :param filepath_pair:必须是这个顺序，这里区分先后顺序
                    """
            syn_path, label_path = filepath_pair
            y1 = librosa.load(syn_path)[0].reshape(-1, 1)
            y2 = librosa.load(label_path)[0].reshape(-1, 1)
            f1 = y1.T  # (feature_dim,seq_len)
            f2 = y2.T  # (feature_dim,seq_len)
            # 使用fastdtw对齐两个MFCC特征矩阵
            _, path = fastdtw(f1.T, f2.T)
            # 对齐后的特征矩阵
            aligned_syn = f1[:, [p[0] for p in path]].T
            aligned_label = f2[:, [p[1] for p in path]].T
            eps = np.finfo(float).eps  # 就是一个保证不为0的极小量
            alpha = np.dot(aligned_syn.T, aligned_label) / (np.dot(aligned_syn.T, aligned_syn) + eps)

            molecular = ((alpha * aligned_label) ** 2).sum()  # 分子
            denominator = ((alpha * aligned_label - aligned_syn) ** 2).sum()  # 分母

            return 10 * np.log10(molecular / (denominator + eps))

        return np.mean(np.array(list(map(cal_pair, self.syn2label.items()))[:10]))

    def compute_f0_corr(self):
        def cal_pair(filepath_pair):
            aligned_f0_1, aligned_f0_2 = self.__get_align_f0(filepath_pair)
            f0corr = np.corrcoef(aligned_f0_1, aligned_f0_2)[0, 1]
            return f0corr

        return np.mean(np.array(list(map(cal_pair, self.syn2label.items()))))

    def compute_f0_rmse(self):
        def pad_to(x, target_len):
            pad_len = target_len - len(x)

            if pad_len <= 0:
                return x[:target_len]
            else:
                return np.pad(x, (0, pad_len), 'constant', constant_values=(0, 0))

        def cal_pair(filepath_pair):
            aligned_f0_1, aligned_f0_2 = self.__get_align_f0(filepath_pair)

            # only calculate f0 error for voiced frame
            y = 1200 * np.abs(np.log2(aligned_f0_1) - np.log2(aligned_f0_2))
            # print(y.sum(), tp_mask.sum())
            f0_rmse_mean = np.mean(y)
            # print(min_cost.shape)
            return f0_rmse_mean

        return np.mean(np.array(list(map(cal_pair, self.syn2label.items()))))

    def compute_log_f0(self):
        def cal_pair(filepath_pair):
            f0_1 = self.__get_mgc(file_path=filepath_pair[0])
            f0_2 = self.__get_mgc(file_path=filepath_pair[1])

            # print(min(len(f0_1), len(f0_2)))
            def logf0_rmse(x, y):
                log_spec_dB_const = 1 / min(len(f0_1), len(f0_2))
                diff = x - y
                return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

            min_cost, _ = librosa.sequence.dtw(f0_1[:, 1:].T, f0_2[:, 1:].T, metric=logf0_rmse)
            # print(min_cost.shape)
            return np.mean(min_cost)

        return np.mean(np.array(list(map(cal_pair, self.syn2label.items()))))

    def compute_ssim(self):
        SAMPLING_RATE = self.SAMPLING_RATE
        ssim_pair_cache = list()
        max_list = list()
        min_list = list()

        def find_ssim_max(audio_file1, audio_file2):
            # 提取MFCC特征
            mfcc1 = librosa.feature.mfcc(y=librosa.load(audio_file1)[0], sr=SAMPLING_RATE)
            mfcc2 = librosa.feature.mfcc(y=librosa.load(audio_file2)[0], sr=SAMPLING_RATE)
            # 使用fastdtw对齐两个MFCC特征矩阵
            _, path = fastdtw(mfcc1.T, mfcc2.T)
            # 对齐后的特征矩阵
            aligned_mfcc1 = mfcc1[:, [p[0] for p in path]].T
            aligned_mfcc2 = mfcc2[:, [p[1] for p in path]].T
            # 归一化对齐后的MFCC特征
            aligned_mfcc1 = aligned_mfcc1 / np.linalg.norm(aligned_mfcc1, axis=0)
            aligned_mfcc2 = aligned_mfcc2 / np.linalg.norm(aligned_mfcc2, axis=0)
            max_ = max(np.max(aligned_mfcc1), np.max(aligned_mfcc2))
            min_ = min(np.min(aligned_mfcc1), np.min(aligned_mfcc2))
            max_list.append(max_)
            min_list.append(min_)
            ssim_pair_cache.append((aligned_mfcc1, aligned_mfcc2))

        for audio_file1, audio_file2 in self.syn2label.items():
            find_ssim_max(audio_file1, audio_file2)

        ssim = StructuralSimilarityIndexMeasure(data_range=max(max_list) - min(min_list))

        def cal_ssim(pair):
            return ssim(
                torch.unsqueeze(torch.unsqueeze(torch.from_numpy(pair[0]), 0), 0),
                torch.unsqueeze(torch.unsqueeze(torch.from_numpy(pair[1]), 0), 0)
            )

        return np.mean(np.array(list(map(cal_ssim, ssim_pair_cache))))

    def compute_mcd24(self):
        # SAMPLING_RATE = 22050
        # FRAME_PERIOD = 5.0
        alpha = 0.435  # 0.65 commonly used at 22050 Hz  #0.44
        fft_size = 512
        mcep_size = 24

        def log_spec_dB_dist(x, y):
            log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
            diff = x - y
            return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

        def wav2mcep_numpy(wavfile, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size, type=None):
            wav, _ = librosa.load(wavfile, sr=self.SAMPLING_RATE, mono=True)
            # Use WORLD vocoder to spectral envelope
            _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=self.SAMPLING_RATE,
                                         frame_period=self.FRAME_PERIOD, fft_size=fft_size)
            # Extract MCEP features
            mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                                   etype=1, eps=1.0E-8, min_det=0.0, itype=3)
            os.makedirs(os.path.join(self.mel_npy_path, "raw"), exist_ok=True)
            os.makedirs(os.path.join(self.mel_npy_path, "syn"), exist_ok=True)
            if "raw" in str(wavfile):
                mcep_name = os.path.join(self.mel_npy_path, "raw", str(wavfile).split("/")[-1].replace(".wav", ""))
            else:
                mcep_name = os.path.join(self.mel_npy_path, "syn", str(wavfile).split("/")[-1].replace(".wav", ""))
            np.save(mcep_name + '.npy', mgc, allow_pickle=False)
            mcep_name = mcep_name + '.npy'
            return mcep_name

        def average_mcd(syn_mcep_file, raw_mcep_file, cost_function):
            # min_cost_tot = 0.0
            # frames_tot = 0
            synth_vec = np.load(syn_mcep_file)
            ref_vec = np.load(raw_mcep_file)  # load MCEP vectors
            ref_frame_no = len(ref_vec)
            # dynamic time warping using librosa
            min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T,
                                               metric=cost_function)
            # min_cost_tot += np.mean(min_cost)
            # frames_tot += ref_frame_no
            # mean_mcd = min_cost_tot / frames_tot
            return min_cost, ref_frame_no

            # print(match_files_dit.items())

        mcd_mean = 0.0
        frames_used_toal = 0
        for key, value in self.syn2label.items():
            syn_mcep_files = wav2mcep_numpy(key, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size, type=None)
            raw_mcep_files = wav2mcep_numpy(value, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size, type=None)
            cost_function = log_spec_dB_dist
            mcd, frames_used = average_mcd(syn_mcep_files, raw_mcep_files, cost_function)
            mcd_mean += np.mean(mcd)
            frames_used_toal += frames_used
        MCD = mcd_mean / frames_used_toal
        return MCD  # , frames_used_toal

    def compute_mcd(self):
        from pymcd.mcd import Calculate_MCD
        def cal_pair(filepath_pair):
            # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
            mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
            return mcd_toolbox.calculate_mcd(filepath_pair[0], filepath_pair[1])

        return np.mean(np.array(list(map(cal_pair, self.syn2label.items()))))

    def __compute_precision(self, feature_type):
        target_list = list(self.syn2label.values())
        generate_list = list(self.syn2label.keys())
        precision_cal_tool = CalPrecision(generate_filepath_list=generate_list, target_filepath_list=target_list)
        if self.clear_cache:
            precision_cal_tool.clear_cache()
        return precision_cal_tool(feature_type)

    def compute_precision_mel(self):
        return self.__compute_precision("mel")

    def compute_precision_mfcc(self):
        return self.__compute_precision("mfcc")

    def __compute_recall(self, feature_type):
        if self.data_type == "VCTK":
            target_list = joblib.load("/your/recall_file_list.joblib")
        elif self.data_type == "LJSpeech":
            target_list = joblib.load("/your/recall_file_list_lj.joblib")
        else:
            raise NotImplementedError
        generate_list = list(self.syn2label.keys())
        recall_cal_tool = CalRecall(generate_filepath_list=generate_list, target_filepath_list=target_list)
        if self.clear_cache:
            recall_cal_tool.clear_cache()
        return recall_cal_tool(feature_type)

    def compute_recall_mfcc(self):
        return self.__compute_recall("mfcc")

    def compute_recall_mel(self):
        return self.__compute_recall("mel")

    def __compute_fid(self, feature_type):
        target_list = list(self.syn2label.values())
        generate_list = list(self.syn2label.keys())
        fid_cal_tool = CalFidSeries(generate_filepath_list=generate_list, target_filepath_list=target_list)
        if self.clear_cache:
            fid_cal_tool.clear_cache()
        return fid_cal_tool(feature_type)

    def compute_fid_mfcc(self):
        return self.__compute_fid("mfcc")

    def compute_fid_mfcc_un_norm(self):
        return self.__compute_fid("mfcc_un_norm")

    def compute_fid_mel(self):
        return self.__compute_fid("mel")

    def __mos_init(self):
        """
        实现mos工具用了才创建，且只创建一次
        :return:
        """
        if self.mos_tool is None:
            self.mos_tool = MOSCal(sample_rate=self.SAMPLING_RATE)

    def __get_file_list_mean_mos(self, filename_list, mos_type="mb"):
        self.__mos_init()
        if mos_type == "mb":
            return np.mean(np.array(list(map(self.mos_tool.get_mb_mos, filename_list))))
        elif mos_type == "ld":
            return np.mean(np.array(list(map(self.mos_tool.get_ld_mos, filename_list))))
        else:
            raise NotImplementedError

    def compute_mb_mos(self):
        syn_list = list(self.syn2label.keys())
        return self.__get_file_list_mean_mos(syn_list, mos_type="mb")

    def compute_ld_mos(self):
        syn_list = list(self.syn2label.keys())
        return self.__get_file_list_mean_mos(syn_list, mos_type="ld")

    def get_target_mos(self, mos_type):
        target_list = list(self.syn2label.values())
        return self.__get_file_list_mean_mos(target_list, mos_type=mos_type)

    def compute_ffe(self):
        # SAMPLING_RATE = 22050
        trim_top_db = 23
        filter_length = 1024
        hop_length = 256

        def load_audio(wav_path):
            wav_raw, _ = librosa.load(wav_path, sr=self.SAMPLING_RATE)
            _, index = librosa.effects.trim(wav_raw, top_db=trim_top_db, frame_length=filter_length,
                                            hop_length=hop_length)
            wav = wav_raw[index[0]:index[1]]
            duration = (index[1] - index[0]) / hop_length
            return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

        ffe = FFE(self.SAMPLING_RATE)
        frame_error_rate = []
        for key, value in self.syn2label.items():
            synth_wav_raw, synth_wav, synth_duration = load_audio(key)
            ref_wav_raw, ref_wav, ref_duration = load_audio(value)
            data = [ref_wav, synth_wav]
            data = pad_1D(data)
            ref_wav, synth_wav = data
            score = ffe.calculate_ffe(torch.tensor(ref_wav), torch.tensor(synth_wav))
            # print(score)
            frame_error_rate.append(score)
        return np.mean(frame_error_rate)  # , np.var(frame_error_rate)

    def compute_speaker_cos(self):
        def get_speaker_cos(filepath_pair):
            def get_speaker_ebd(wav_filepath):
                def wav_to_16000(wav_filepath):
                    if osp.exists(wav_filepath + "_16000"):
                        return wav_filepath + "_16000"
                    else:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_wav(wav_filepath)
                        audio = audio.set_frame_rate(16000)
                        audio.export(wav_filepath + "_16000")
                        return wav_filepath + "_16000"

                fpath = Path(wav_to_16000(wav_filepath))
                wav = preprocess_wav(fpath)

                encoder = VoiceEncoder()
                return encoder.embed_utterance(wav)

            return np.mean(cosine_similarity(
                get_speaker_ebd(filepath_pair[0]).reshape(1, -1),
                get_speaker_ebd(filepath_pair[1]).reshape(1, -1)
            ))

        return np.mean(np.array(list(map(get_speaker_cos, self.syn2label.items()))))

    def compute_speaker_cos_direct(self):
        def get_speaker_cos(filepath_pair):
            def get_speaker_ebd(wav_filepath):
                fpath = Path(wav_filepath)
                wav = preprocess_wav(fpath)

                encoder = VoiceEncoder()
                return encoder.embed_utterance(wav)

            return np.mean(cosine_similarity(
                get_speaker_ebd(filepath_pair[0]).reshape(1, -1),
                get_speaker_ebd(filepath_pair[1]).reshape(1, -1)
            ))

        return np.mean(np.array(list(map(get_speaker_cos, self.syn2label.items()))))

    def compute_mfcc_cos(self):
        def get_pair_mfcc_cos(filepath_pair):
            mfcc1 = librosa.feature.mfcc(y=librosa.load(filepath_pair[0])[0], sr=self.SAMPLING_RATE)
            mfcc2 = librosa.feature.mfcc(y=librosa.load(filepath_pair[1])[0], sr=self.SAMPLING_RATE)
            # 使用fastdtw对齐两个MFCC特征矩阵
            _, path = fastdtw(mfcc1.T, mfcc2.T)
            # 对齐后的特征矩阵
            aligned_mfcc1 = mfcc1[:, [p[0] for p in path]].T
            aligned_mfcc2 = mfcc2[:, [p[1] for p in path]].T
            # 归一化对齐后的MFCC特征
            aligned_mfcc1 = aligned_mfcc1 / np.linalg.norm(aligned_mfcc1, axis=0)
            aligned_mfcc2 = aligned_mfcc2 / np.linalg.norm(aligned_mfcc2, axis=0)
            return cosine_similarity(
                aligned_mfcc1.reshape(1, -1),
                aligned_mfcc2.reshape(1, -1)
            )

        return np.mean(np.array(list(map(get_pair_mfcc_cos, self.syn2label.items()))))

    def compute_mel_sdr(self):
        def calculate_sdr(file_pair):
            distorted_file, original_file = file_pair

            def compute_mel(wav_filepath):
                """
                这里本身不应该实现的cache方式，但是由于环境问题智能在这里实现了。
                :param wav_filepath:
                :return:
                """

                def get_cache_filepath():
                    wav_dir = osp.dirname(wav_filepath)
                    cache_dir = osp.join(wav_dir + "_mel")
                    base_name = osp.basename(wav_filepath) + ".npy"
                    os.makedirs(cache_dir, exist_ok=True)
                    return osp.join(cache_dir, base_name)

                cache_filepath = get_cache_filepath()
                if osp.exists(cache_filepath):
                    mel_spectrogram = np.load(cache_filepath)
                else:
                    raise NotImplementedError
                return mel_spectrogram  # (seq_len,80)

            eps = np.finfo(float).eps  # 就是一个保证不为0的极小量
            f1 = compute_mel(original_file).T  # (feature_dim,seq_len)
            f2 = compute_mel(distorted_file).T  # (feature_dim,seq_len)
            # 使用fastdtw对齐两个MFCC特征矩阵
            _, path = fastdtw(f1.T, f2.T)
            # 对齐后的特征矩阵
            aligned_f1 = f1[:, [p[0] for p in path]].T
            aligned_f2 = f2[:, [p[1] for p in path]].T

            # 确保两个信号具有相同的长度

            original = aligned_f1
            distorted = aligned_f2

            # 计算失真信号
            distortion = distorted - original + eps

            # 计算SDR
            sdr = 10 * np.log10(np.sum(original ** 2) / np.sum(distortion ** 2))

            return sdr

        return np.mean(np.array(list(map(calculate_sdr, self.syn2label.items()))))

    def compute_mfcc_e_cos(self):
        def get_pair_mfcc_cos(filepath_pair):
            mfcc1 = librosa.feature.mfcc(y=librosa.load(filepath_pair[0])[0], sr=self.SAMPLING_RATE)
            mfcc2 = librosa.feature.mfcc(y=librosa.load(filepath_pair[1])[0], sr=self.SAMPLING_RATE)
            # 使用fastdtw对齐两个MFCC特征矩阵
            _, path = fastdtw(mfcc1.T, mfcc2.T)
            # 对齐后的特征矩阵
            aligned_mfcc1 = mfcc1[:, [p[0] for p in path]].T
            aligned_mfcc2 = mfcc2[:, [p[1] for p in path]].T  # (seq_len,20)
            # 归一化对齐后的MFCC特征
            aligned_mfcc1 = aligned_mfcc1 / np.linalg.norm(aligned_mfcc1, axis=0)
            aligned_mfcc2 = aligned_mfcc2 / np.linalg.norm(aligned_mfcc2, axis=0)
            cos_list = list()
            for i in range(len(aligned_mfcc1)):
                cos_list.append(cosine_similarity(
                    aligned_mfcc1[i].reshape(1, -1),
                    aligned_mfcc2[i].reshape(1, -1)
                ))
            return np.mean(np.array(cos_list))

        return np.mean(np.array(list(map(get_pair_mfcc_cos, self.syn2label.items()))))

    def compute_deep_speaker_cos(self):
        # SAMPLING_RATE = 22050
        trim_top_db = 23
        filter_length = 1024
        hop_length = 256
        arg_dit = {"sampling_rate": self.SAMPLING_RATE,
                   "win_length": 1024,
                   "speaker_embedder": "DeepSpeaker",
                   "speaker_embedder_cuda": False}

        def load_audio(wav_path):
            wav_raw, _ = librosa.load(wav_path, sr=self.SAMPLING_RATE)
            _, index = librosa.effects.trim(wav_raw, top_db=trim_top_db, frame_length=filter_length,
                                            hop_length=hop_length)
            wav = wav_raw[index[0]:index[1]]
            duration = (index[1] - index[0]) / hop_length
            return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

        args = argparse.Namespace(**arg_dit)
        speaker_emb = PreDefinedEmbedder(args)
        cosine_score = []
        for key, value in self.syn2label.items():
            synth_wav_raw, synth_wav, synth_duration = load_audio(key)
            synth_spker_embed = speaker_emb(synth_wav)
            ref_wav_raw, ref_wav, ref_duration = load_audio(value)
            ref_spker_embed = speaker_emb(ref_wav)
            score = cosine_similarity(ref_spker_embed, synth_spker_embed)
            cosine_score.append(score)
        return np.mean(np.array(cosine_score))

    def get_all(self):
        """
        :return:
        """
        mcd = self.compute_mcd()
        ssim = self.compute_ssim()
        ffe = self.compute_ffe()
        return ssim, ffe, mcd

    def get_metrics_by_list(self, metrics_list=None):
        """
        """
        if metrics_list is None:
            metrics_list = ["ssim", "ffe", "mcd"]

        return list(map(
            lambda metrics_name:
            getattr(self, "compute_" + metrics_name)(),
            metrics_list))


class CalOneModel:
    def __init__(self, folder_path, raw_folder=None, data_type=None, clear_cache=False, file_find_type=None):
        """
        """
        self.folder_path = folder_path
        self.file_find_type = file_find_type
        self.subdirectories = self.__get_subdirectories()
        self.raw_folder = raw_folder
        if data_type is None:
            self.data_type = "VCTK"
        else:
            self.data_type = data_type
        self.clear_cache = clear_cache

    def __get_subdirectories(self):
        """
        :param:
        :return:
        """
        subdirectories = []
        if self.file_find_type is None:
            for item in os.listdir(self.folder_path):
                item_path = osp.join(self.folder_path, item)
                if os.path.isdir(item_path) and item.isdigit():  # 后面这个判据，是为了消除mel缓存的影响
                    subdirectories.append(os.path.join(self.folder_path, item))
        elif self.file_find_type == "key_step":
            for i in [100000, 200000, 300000]:
                subdirectories.append(osp.join(self.folder_path, str(i)))
        elif self.file_find_type == "only_end":
            subdirectories.append(osp.join(self.folder_path, str(300000)))
        else:
            raise NotImplementedError
        return subdirectories

    def get_model_metrics_ssim_ffe_mcd(self):
        ssim_ffe_mcd_list = list()
        for syn_dir in self.subdirectories:
            cal_tool = Cal(syn_folder=syn_dir)
            ssim, ffe, mcd = cal_tool.get_all()
            ssim_ffe_mcd_list.append((int(osp.basename(syn_dir)), ssim, ffe, mcd))

            # 每次都保存避免中间出错全丢了
            ssim_ffe_mcd_df = pd.DataFrame(ssim_ffe_mcd_list)
            ssim_ffe_mcd_df.columns = ["train_step", "ssim", "ffe", "mcd"]
            ssim_ffe_mcd_df = ssim_ffe_mcd_df.sort_values("train_step").reset_index(drop=True)
            ssim_ffe_mcd_df.to_csv(osp.join(self.folder_path, "metrics.csv"))

    def get_model_metrics_by_list(self, metrics_name_list):
        """

        :param metrics_name_list: ["ssim", "ffe", "mcd", "speaker_cos", "fid_mfcc", "fid_mel"]
        :return:
        """
        metrics_list = list()
        metrics_name = "_".join(metrics_name_list)
        if self.file_find_type is not None:
            metrics_name = self.file_find_type + metrics_name
        if os.path.exists(osp.join(self.folder_path, "metrics_" + metrics_name + ".csv")):
            metrics_cache = pd.read_csv(osp.join(self.folder_path, "metrics_" + metrics_name + ".csv"))
        else:
            metrics_cache = None

        for syn_dir in self.subdirectories:
            if metrics_cache is not None and int(osp.basename(syn_dir)) in metrics_cache["train_step"]:
                metrics = list(metrics_cache[metrics_cache["train_step"] == int(syn_dir)][metrics_name_list])
            else:
                if self.raw_folder is not None:
                    cal_tool = Cal(syn_folder=syn_dir, data_type=self.data_type, raw_folder=self.raw_folder)
                else:
                    cal_tool = Cal(syn_folder=syn_dir, data_type=self.data_type)
                metrics = cal_tool.get_metrics_by_list(metrics_name_list)
            metrics_list.append((int(osp.basename(syn_dir)), *metrics))

            # 每次都保存避免中间出错全丢了
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.columns = ["train_step", *metrics_name_list]
            metrics_df = metrics_df.sort_values("train_step").reset_index(drop=True)
            metrics_df.to_csv(osp.join(self.folder_path, "metrics_" + metrics_name + ".csv"))


if __name__ == "__main__":
    eval_dir_list = list([
        "your/result/VCTK_cm",
    ])
    for eval_dir in eval_dir_list:
        if eval_dir in [
            "your/result/VCTK_cm1"
        ]:
            cal_one_model_tool = CalOneModel(eval_dir)
        elif eval_dir in ["your/result/VCTK_cm1"]:
            cal_one_model_tool = CalOneModel(eval_dir, file_find_type="only_end")
        else:
            cal_one_model_tool = CalOneModel(eval_dir, file_find_type="key_step")
        # cal_one_model_tool.get_model_metrics_by_list(["fid_align_mel", "fid_align_mfcc_un_norm"])

        # cal_one_model_tool.get_model_metrics_by_list(["f0_rmse"])
        # cal_one_model_tool.get_model_metrics_by_list(["f0_corr"])
        # cal_one_model_tool.get_model_metrics_by_list(["mcd"])
        # cal_one_model_tool.get_model_metrics_by_list(["recall_mfcc"])
        # cal_one_model_tool.get_model_metrics_by_list(["speaker_cos"])
        cal_one_model_tool.get_model_metrics_by_list(["wer_un_comma"])
        # cal_one_model_tool.get_model_metrics_by_list(["ffe", "ssim","mfcc_cos"])
