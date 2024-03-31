import os.path as osp
import os
from typing import List

import joblib
import librosa
import numpy as np
from scipy import linalg
import audio as Audio
from fastdtw import fastdtw


class CalFeature:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.all_feature_type = ["mfcc", "mel", "mfcc_un_norm"]
        self.feature_type2dim_dict = dict(
            mfcc=20, mel=80, mfcc_un_norm=20,
        )
        self.cal_type = ""  # 下面类需要重载该属性

    def clear_cache(self):
        for feature_type in self.all_feature_type:
            for target_or_generate in ["target", "generate"]:
                cache_filepath = self.get_cache_path(feature_type=feature_type, target_or_generate=target_or_generate)
                if osp.exists(cache_filepath):
                    # 删除文件
                    os.remove(cache_filepath)

    def get_cache_path(self, feature_type, target_or_generate):
        deal_filepath_list = getattr(self, target_or_generate + "_filepath_list")
        return osp.join(
            osp.dirname(deal_filepath_list[0]),
            target_or_generate + "_" + feature_type + "_" + self.cal_type + "_cache.joblib"
        )

    def compute_mfcc(self, wav_filepath):
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=librosa.load(wav_filepath)[0], sr=self.sample_rate).T  # (seq_len,20)

        # 归一化对齐后的MFCC特征
        return mfcc / np.linalg.norm(mfcc, axis=0)  # (seq_len,20)

    def compute_mfcc_un_norm(self, wav_filepath):
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=librosa.load(wav_filepath)[0], sr=self.sample_rate).T  # (seq_len,20)
        return mfcc  # (seq_len,20)

    def compute_mel(self, wav_filepath):
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
            # 注意下面的参数要和您在数据处理部分的参数一致
            STFT = Audio.stft.TacotronSTFT(
                filter_length=1024,
                hop_length=256,
                win_length=1024,
                n_mel_channels=80,
                sampling_rate=self.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
            )
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(librosa.load(wav_filepath)[0], STFT)
            mel_spectrogram = mel_spectrogram.T
            np.save(cache_filepath, mel_spectrogram)
        # 转置之后返回
        return mel_spectrogram  # (seq_len,80)

    @staticmethod
    def manifold_estimate(A_features, B_features, k):
        """
        部分并行的运算方式，降低了内存开销，适合大部分的情况。
        如果您对运算速度不满意，且内存还有剩余，您可以适当调整拆分方法，用内存来换计算时间
        :param A_features:
        :param B_features:
        :param k:
        :return:
        """
        a_len_sqrt = int(np.sqrt(len(A_features)))
        kth_smallest_dis_list = list()
        for i, j in zip(range(a_len_sqrt), range(a_len_sqrt-1, -1, -1)):
            if j != 0:
                mini_distances_a = np.linalg.norm(
                    A_features[i * a_len_sqrt:(i + 1) * a_len_sqrt, np.newaxis, :] - A_features,
                    axis=2)
            else:
                mini_distances_a = np.linalg.norm(
                    A_features[i * a_len_sqrt:, np.newaxis, :] - A_features,
                    axis=2)
            mini_kth_smallest_dis = np.partition(mini_distances_a, k, axis=1)[:, k]
            del mini_distances_a
            kth_smallest_dis_list.append(mini_kth_smallest_dis)
        kth_smallest_dis = np.concatenate(kth_smallest_dis_list)
        count = 0
        b_len_sqrt = int(np.sqrt(len(B_features)))
        for i in range(a_len_sqrt+1):
            mini_distances_b2a = np.linalg.norm(
                B_features[i * b_len_sqrt:(i + 1) * b_len_sqrt, np.newaxis, :] - A_features, axis=2)
            mini_result = mini_distances_b2a - kth_smallest_dis[np.newaxis, :]
            count += int(np.sum(np.any(mini_result <= 0, axis=1)))
        return count / len(B_features)

    @staticmethod
    def manifold_estimate_fully_parallel(A_features, B_features, k):
        """
        这种实现方法是全并行的，会给内存带来巨大压力，请慎重使用。
        您应该在运行的同时同步查看内存开销，查看内存开销可以用： free -h
        用a特征去覆盖b特征，
        :param A_features: np.ndarray(n,feature_dim)
        :param B_features: np.ndarray(n,feature_dim)
        :param k:
        :return:
        """
        distances_a = np.linalg.norm(A_features[:, np.newaxis, :] - A_features, axis=2)
        fourth_smallest = np.partition(distances_a, k, axis=1)[:, k]
        del distances_a
        distances_b2a = np.linalg.norm(B_features[:, np.newaxis, :] - A_features, axis=2)
        result = distances_b2a - fourth_smallest[np.newaxis, :]
        del distances_b2a
        del fourth_smallest
        count = np.sum(np.any(result <= 0, axis=1))
        return count / len(B_features)

    @staticmethod
    def manifold_estimate_fully_sequential(A_features, B_features, k):
        """
        当前方案是全串行，运行速度较慢，请谨慎使用
        :param A_features:
        :param B_features:
        :param k:
        :return:
        """
        KNN_list_in_A = {}
        for i, A in enumerate(A_features):
            pairwise_distances = np.zeros(shape=(len(A_features)))

            for j, A_prime in enumerate(A_features):
                d = np.linalg.norm((A - A_prime), ord=2)
                pairwise_distances[j] = d

            v = np.partition(pairwise_distances, k)[k]
            KNN_list_in_A[i] = v

        n = 0

        for B in B_features:
            for i, A_prime in enumerate(A_features):
                d = np.linalg.norm((B - A_prime), ord=2)
                if d <= KNN_list_in_A[i]:
                    n += 1
                    break  # 找到一个覆盖就不再寻找下一个

        return n / len(B_features)


class CalFidSeries(CalFeature):
    # 这里是定义一个如何计算TTS FID类似的函数体
    # 主要是定义一下各种不同的特征的获取再最终用同一个计算逻辑接住
    def __init__(self, generate_filepath_list, target_filepath_list, sample_rate=22050):
        """

        @param generate_filepath_list: 生成数据组成的list
        @param target_filepath_list: 训练数据组成的list
        """
        super().__init__(sample_rate=sample_rate)
        self.cal_type = "fid"
        self.SAMPLING_RATE = sample_rate
        self.generate_filepath_list: List = generate_filepath_list
        self.target_filepath_list = target_filepath_list

    def __call__(self, feature_type):
        assert feature_type in self.all_feature_type
        return self.__get_fid_by_single(feature_type)

    def __get_fid_by_single(self, feature_type, ):
        """

        @param feature_type: 当前处理何种特征
        @return: target_f_mean, target_f_cov, generate_f_mean, generate_f_cov
        """

        def get_f_mean_cov_by_single(target_or_generate):
            """
            处理数据的读入和特征提取，如果有cache就cache一下
            @param target_or_generate: str in ["target", "generate"]
            @return:
            """
            assert target_or_generate in ["target", "generate"]
            deal_filepath_list = getattr(self, target_or_generate + "_filepath_list")
            cache_path = self.get_cache_path(feature_type=feature_type, target_or_generate=target_or_generate)
            if osp.exists(cache_path):
                cache_dict = joblib.load(cache_path)
                feature_mean = cache_dict["mu"]
                feature_cov = cache_dict["sigma"]
            else:
                feature_list = list()
                for filepath in deal_filepath_list:
                    deal_fun = getattr(self, "compute_" + feature_type)
                    feature_list.append(deal_fun(filepath))
                feature = np.concatenate(feature_list, axis=0)
                feature_mean = np.mean(feature, axis=0)
                feature_cov = np.cov(feature, rowvar=False)
                cache_dict = dict(
                    mu=feature_mean,
                    sigma=feature_cov,
                )
                joblib.dump(cache_dict, cache_path)

            return feature_mean, feature_cov

        # 处理原始数据的特征
        target_f_mean, target_f_cov = get_f_mean_cov_by_single("target")
        # 处理生成数据的特征
        generate_f_mean, generate_f_cov = get_f_mean_cov_by_single("generate")
        return self.__calculate_frechet_distance(
            mu1=target_f_mean,
            mu2=generate_f_mean,
            sigma1=target_f_cov,
            sigma2=generate_f_cov,
        )

    @staticmethod
    def __calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        这个函数都是最后计算的那个共有部分
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


class CalFIDAlign(CalFeature):
    def __init__(self, generate_filepath_list, target_filepath_list, sample_rate=22050):
        """

        @param generate_filepath_list: 生成数据组成的list
        @param target_filepath_list: 训练数据组成的list
        """
        # super().__init__(sample_rate=sample_rate)
        super().__init__(sample_rate)
        self.cal_type = "fid"
        self.SAMPLING_RATE = sample_rate
        self.generate_filepath_list: List = generate_filepath_list
        self.target_filepath_list = target_filepath_list

    def __call__(self, feature_type, norm=False, *args, **kwargs):
        return self.__fid(feature_type, norm=norm)

    def __fid(self, feature_type, norm=False):
        assert feature_type in self.all_feature_type
        def get_pair_mfcc(filepath_pair):
            deal_fun = getattr(self, "compute_" + feature_type)
            f1 = deal_fun(filepath_pair[0]).T  # (feature_dim,seq_len)
            f2 = deal_fun(filepath_pair[1]).T  # (feature_dim,seq_len)
            # 使用fastdtw对齐两个MFCC特征矩阵
            _, path = fastdtw(f1.T, f2.T)
            # 对齐后的特征矩阵
            aligned_f1 = f1[:, [p[0] for p in path]].T
            aligned_f2 = f2[:, [p[1] for p in path]].T
            if norm:
                # 归一化对齐后的MFCC特征
                aligned_f1 = aligned_f1 / np.linalg.norm(aligned_f1, axis=0)
                aligned_f2 = aligned_f2 / np.linalg.norm(aligned_f2, axis=0)
            return aligned_f1, aligned_f2
        target_f_list = list()
        generate_f_list = list()
        for target_generate_path_pair in zip(self.target_filepath_list, self.generate_filepath_list):
            target_mfcc, generate_mfcc = get_pair_mfcc(target_generate_path_pair)
            target_f_list.append(target_mfcc)
            generate_f_list.append(generate_mfcc)
        target_f = np.concatenate(target_f_list, axis=0)
        generate_f = np.concatenate(generate_f_list, axis=0)

        target_f_mean = np.mean(target_f, axis=0)
        target_f_cov = np.cov(target_f, rowvar=False)

        generate_f_mean = np.mean(generate_f, axis=0)
        generate_f_cov = np.cov(generate_f, rowvar=False)

        return self.__calculate_frechet_distance(
            target_f_mean,
            target_f_cov,
            generate_f_mean,
            generate_f_cov
        )

    @staticmethod
    def __calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        这个函数都是最后计算的那个共有部分
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)



class CalRecall(CalFeature):
    def __init__(self, generate_filepath_list, target_filepath_list, sample_rate=22050, k=3):
        super().__init__(sample_rate)
        self.cal_type = "recall"
        self.generate_filepath_list: List = generate_filepath_list
        self.target_filepath_list = target_filepath_list
        self.k = k

    def __call__(self, feature_type):
        assert feature_type in self.all_feature_type
        return self.__get_recall(feature_type)

    def __get_recall(self, feature_type):
        def get_f_by_single(target_or_generate):
            """
            处理数据的读入和特征提取，如果有cache就cache一下
            @param target_or_generate: str in ["target", "generate"]
            @return:
            """
            assert target_or_generate in ["target", "generate"]
            deal_filepath_list = getattr(self, target_or_generate + "_filepath_list")
            cache_path = self.get_cache_path(feature_type=feature_type, target_or_generate=target_or_generate)
            if osp.exists(cache_path):
                feature = joblib.load(cache_path)
            else:
                feature_list = list()
                for filepath in deal_filepath_list:
                    deal_fun = getattr(self, "compute_" + feature_type)
                    feature_list.append(deal_fun(filepath))
                feature = np.concatenate(feature_list, axis=0)
                joblib.dump(feature, cache_path)

            return feature

        # 处理原始数据的特征
        target_f = get_f_by_single("target")
        # 处理生成数据的特征
        generate_f = get_f_by_single("generate")

        return self.manifold_estimate(generate_f, target_f, self.k)


class CalPrecision(CalFeature):
    def __init__(self, generate_filepath_list, target_filepath_list, sample_rate=22050, k=3):
        super().__init__(sample_rate)
        self.cal_type = "recall"
        self.generate_filepath_list: List = generate_filepath_list
        self.target_filepath_list = target_filepath_list
        self.k = k

    def __call__(self, feature_type):
        assert feature_type in self.all_feature_type
        return self.__get_precision(feature_type)

    def __get_precision(self, feature_type):
        def get_f_by_single(target_or_generate):
            """
            处理数据的读入和特征提取，如果有cache就cache一下
            @param target_or_generate: str in ["target", "generate"]
            @return:
            """
            assert target_or_generate in ["target", "generate"]
            deal_filepath_list = getattr(self, target_or_generate + "_filepath_list")
            cache_path = self.get_cache_path(feature_type=feature_type, target_or_generate=target_or_generate)
            if osp.exists(cache_path):
                feature = joblib.load(cache_path)
            else:
                feature_list = list()
                for filepath in deal_filepath_list:
                    deal_fun = getattr(self, "compute_" + feature_type)
                    feature_list.append(deal_fun(filepath))
                feature = np.concatenate(feature_list, axis=0)
                joblib.dump(feature, cache_path)

            return feature

        # 处理原始数据的特征
        target_f = get_f_by_single("target")
        # 处理生成数据的特征
        generate_f = get_f_by_single("generate")

        return self.manifold_estimate(target_f, generate_f, self.k)


if __name__ == "__main__":
    target_list = list(
        ["D:\毕设\TTS论文\p226-020-lsm.wav"]
    ) * 2
    generate_list = list(
        ["D:\毕设\TTS论文\p226-020-uniform.wav"]
    ) * 2

    fid_cal_tool = CalFidSeries(generate_filepath_list=generate_list, target_filepath_list=target_list)

    print(fid_cal_tool("mfcc"))
    fid_cal_tool.clear_cache()
    fid_cal_tool = CalFidSeries(generate_filepath_list=generate_list, target_filepath_list=generate_list)

    print(fid_cal_tool("mfcc"))
    fid_cal_tool.clear_cache()

    recall_cal_tool = CalRecall(generate_filepath_list=generate_list, target_filepath_list=generate_list)
    print(recall_cal_tool("mfcc"))
    recall_cal_tool.clear_cache()
