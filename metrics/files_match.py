import os
import re
import librosa
import re
import numpy as np
import torch
import pyworld
import pysptk
import math
import argparse
from speakerembedder import PreDefinedEmbedder
from sklearn.metrics.pairwise import cosine_similarity
from f0_frame_error import FFE
from utils.tools import pad_1D
syn_folder = "/data/Code_lx/data1029/DiffGAN-TTS-main/output/result/VCTK_diff_gan_naive/100000"
raw_folder = "/data/Code_lx/data1029/DiffGAN-TTS-main/raw_data/VCTK"
mel_npy_path = "/data/Code_lx/data1029/DiffGAN-TTS-main/output/mel_npy"
match_files = {}
MCD = []
Frames = []
def filematch(syn_folder, raw_folder):
    target_extension = ".wav"
    syn_names = os.listdir(syn_folder)
    filtered_syn_names = [file for file in syn_names if file.endswith(target_extension)]
    for syn_name in filtered_syn_names:
        syn_raw_file = os.path.join(syn_folder, syn_name) 
        raw_folder_name = str(syn_name).split("-")[0]
        raw_wav_file = os.path.join(raw_folder, raw_folder_name, syn_name) 
        match_files.update({syn_raw_file: raw_wav_file}) 
          

def average_frame_error_rate(match_files_dit):
    SAMPLING_RATE = 22050
    FRAME_PERIOD = 5.0
    trim_top_db = 23
    filter_length = 1024
    hop_length = 256
    def load_audio(wav_path):
        wav_raw, _ = librosa.load(wav_path, sr=SAMPLING_RATE)
        _, index = librosa.effects.trim(wav_raw, top_db= trim_top_db, frame_length= filter_length, hop_length= hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration) 
    ffe = FFE(SAMPLING_RATE) 
    frame_error_rate = []
    for key, value in match_files_dit.items():
      synth_wav_raw, synth_wav, synth_duration = load_audio(key)
      ref_wav_raw, ref_wav, ref_duration = load_audio(value)
      data = [ref_wav,synth_wav]
      data = pad_1D(data)
      ref_wav,synth_wav = data
      score = ffe.calculate_ffe(torch.tensor(ref_wav),torch.tensor(synth_wav))
    # print(score)
      frame_error_rate.append(score)
    return np.mean(frame_error_rate),np.var(frame_error_rate)


filematch(syn_folder, raw_folder)
results = average_frame_error_rate(match_files)
print(results)










