import os
import math
import glob
import librosa
import pyworld
import pysptk
import numpy as np
import matplotlib.pyplot as plot

syn_folder = "/data/Code_lx/data1029/DiffGAN-TTS-main/output/result/VCTK_diff_gan_naive/300000"
raw_folder = "/data/Code_lx/data1029/DiffGAN-TTS-main/raw_data/VCTK"
mel_npy_path = "/data/Code_lx/data1029/DiffGAN-TTS-main/output/mel_npy"
match_files = {}

def filematch(syn_folder, raw_folder):
    target_extension = ".wav"
    syn_names = os.listdir(syn_folder)
    filtered_syn_names = [file for file in syn_names if file.endswith(target_extension)]
    for syn_name in filtered_syn_names:
        syn_raw_file = os.path.join(syn_folder, syn_name) 
        raw_folder_name = str(syn_name).split("-")[0]
        raw_wav_file = os.path.join(raw_folder, raw_folder_name, syn_name) 
        match_files.update({syn_raw_file: raw_wav_file}) 
        
SAMPLING_RATE = 22050
num_mcep = 24
frame_period = 5.0
n_frames = 128

def F0_RMSE_CAL(match_files_dit):
    min_cost_tot= 0
    cost_tot = []
    fram_tot = 0
    SAMPLING_RATE = 22050
    FRAME_PERIOD = 5.0
    alpha = 0.435  # 0.65 commonly used at 22050 Hz  #0.44
    fft_size = 512
    mcep_size = 24
    def logf0_rmse(x, y):
        # log_spec_dB_const = 1/len(frame_len)
        log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = x - y
        return log_spec_dB_const * math.sqrt(np.inner(diff, diff))
    cost_function = logf0_rmse
    def wav2mcep_numpy(wavfile, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size,type=None):
        wav, _ = librosa.load(wavfile, sr=SAMPLING_RATE, mono=True)
        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=SAMPLING_RATE,
                            frame_period=FRAME_PERIOD, fft_size=fft_size)
        # Extract MCEP features
        mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                            etype=1, eps=1.0E-8, min_det=0.0, itype=3)  
        if "raw" in str(wavfile):
            mcep_name = os.path.join(mel_npy_path, "raw", str(wavfile).split("/")[-1].replace(".wav",""))
        else:
            mcep_name = os.path.join(mel_npy_path, "syn", str(wavfile).split("/")[-1].replace(".wav",""))       
        np.save(mcep_name + '.npy', mgc, allow_pickle=False)
        mcep_name = mcep_name + '.npy'
        return mcep_name
    for key, value in match_files_dit.items():    
        # syn_wav, _ = librosa.load(key, sr = SAMPLING_RATE, mono = True)
        # raw_wav, _ = librosa.load(value, sr = SAMPLING_RATE, mono = True)
        syn_mcep_file = wav2mcep_numpy(key, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size,type=None)
        raw_mcep_file = wav2mcep_numpy(value, alpha=alpha, fft_size=fft_size, mcep_size=mcep_size,type=None)
        synth_vec = np.load(syn_mcep_file) 
        ref_vec = np.load(raw_mcep_file)
        fram_tot = len(ref_vec)
        min_cost, _ = librosa.sequence.dtw(synth_vec[:].T, ref_vec[:].T, metric=cost_function)  
        min_cost_tot += np.mean(min_cost)
        # break 
        cost_tot.append(min_cost_tot)
        print(cost_tot)
        print(min_cost_tot / fram_tot)
    f0_rmse = min_cost_tot / fram_tot
    return f0_rmse 

#mean_logf0_rmse = min_cost_tot/len(frame_len)
#mean_logf0_rmse
filematch(syn_folder, raw_folder)

Mean_f0_rems = F0_RMSE_CAL(match_files)
print(f'F0_RMSE={Mean_f0_rems}')
