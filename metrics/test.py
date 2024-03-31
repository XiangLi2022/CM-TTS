import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
# 加载两段语音文件
audio_file1 = "/data/Code_lx/data1029/DiffGAN-TTS-main/output/result/VCTK_diff_gan_naive/300000/p226-020.wav"
#audio_file2 = "/data/Code_lx/data1029/DiffGAN-TTS-main/output/result/VCTK_diff_gan_naive/300000/p226-020.wav"
audio_file2 = "/data/Code_lx/data1029/DiffGAN-TTS-main/raw_data/VCTK/p226/p226-020.wav"


def get_ssim(audio_file1,audio_file2):
    # 提取MFCC特征
    mfcc1 = librosa.feature.mfcc(y=librosa.load(audio_file1)[0], sr=22050)
    mfcc2 = librosa.feature.mfcc(y=librosa.load(audio_file2)[0], sr=22050)

    # 使用fastdtw对齐两个MFCC特征矩阵
    _, path = fastdtw(mfcc1.T, mfcc2.T)

    # 对齐后的特征矩阵
    aligned_mfcc1 = mfcc1[:, [p[0] for p in path]].T
    aligned_mfcc2 = mfcc2[:, [p[1] for p in path]].T

    # 归一化对齐后的MFCC特征
    aligned_mfcc1 = aligned_mfcc1 / np.linalg.norm(aligned_mfcc1, axis=0)
    aligned_mfcc2 = aligned_mfcc2 / np.linalg.norm(aligned_mfcc2, axis=0)

    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(
        torch.unsqueeze(torch.unsqueeze(torch.from_numpy(aligned_mfcc1), 0), 0),
        torch.unsqueeze(torch.unsqueeze(torch.from_numpy(aligned_mfcc2), 0), 0)
        ).numpy()

if __name__ == "__main__":
    # print(aligned_mfcc1.shape)
    # print(aligned_mfcc1.flatten().shape)
    # print(aligned_mfcc2.shape)
    # print(aligned_mfcc2.flatten().shape)
    print(get_ssim(audio_file1,audio_file2))
    pass
    

# # # 计算余弦相似度
# # cosine_similarity_score = 1 - cosine(aligned_mfcc1.flatten(), aligned_mfcc2.flatten())



# # print(cosine_similarity_score)

# # # 输出相似度矩阵（cosine_similarity_score），它表示对齐后的MFCC特征矩阵之间的相似度



# # 计算DTW距离
# distance, _ = fastdtw(aligned_mfcc1.T, aligned_mfcc2.T, dist=euclidean)

# print(f"DTW距离: {distance}")
















