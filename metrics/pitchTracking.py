import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

class Pitch:
    def __init__(self, sr=16000):
        self.sr=sr

    def extract_f0_path(self, path):
        y, sr = torchaudio.load(path)
        return self.extract_f0(y)

    def extract_f0(self, y):
        y_f0 = self.compute_yin(y.view(-1), self.sr)
        return y_f0

    def plot_f0_path(self, path, save_path=None):
        sig, sr = torchaudio.load(path, save_path)
        self.plot_f0(sig)

    def plot_f0(self, sig, save_path=None):
        sig = sig.view(-1)
        pitches, harmonic_rates, argmins, times = self.compute_yin(sig, self.sr)
        duration = len(sig)/float(self.sr)
        harmo_thresh=0.1

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
        ax1.set_title('Audio data')
        ax1.set_ylabel('Amplitude')
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
        ax2.set_title('F0')
        ax2.set_ylabel('Frequency (Hz)')
        # ax3 = plt.subplot(3, 1, 3, sharex=ax2)
        # ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], harmonic_rates)
        # ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], [harmo_thresh] * len(harmonic_rates), 'r')
        # ax3.set_title('Harmonic rate')
        # ax3.set_ylabel('Rate')
        # ax4 = plt.subplot(4, 1, 4, sharex=ax2)
        # ax4.plot([float(x) * duration / len(argmins) for x in range(0, len(argmins))], argmins)
        # ax4.set_title('Index of minimums of CMND')
        # ax4.set_ylabel('Frequency (Hz)')
        # ax4.set_xlabel('Time (seconds)')

        # plt.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def compute_yin(self, sig, sr, w_len=512, w_step=256, f0_min=100, f0_max=500,
                harmo_thresh=0.1):
        """
        Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.
        https://github.com/NVIDIA/mellotron adaption of
        https://github.com/patriceguyot/Yin
        :param sig: Audio signal (list of float)
        :param sr: sampling rate (int)
        :param w_len: size of the analysis window (samples)
        :param w_step: size of the lag between two consecutives windows (samples)
        :param f0_min: Minimum fundamental frequency that can be detected (hertz)
        :param f0_max: Maximum fundamental frequency that can be detected (hertz)
        :param harmo_thresh: Threshold of detection. The yalgorithmù return the
        first minimum of the CMND function below this threshold.
        :returns:
            * pitches: list of fundamental frequencies,
            * harmonic_rates: list of harmonic rate values for each fundamental
            frequency value (= confidence value)
            * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
            * times: list of time of each estimation
        :rtype: tuple
        """

        tau_min = int(sr / f0_max)
        tau_max = int(sr / f0_min)

        # time values for each analysis window
        time_scale = range(0, len(sig) - w_len, w_step)
        times = [t/float(sr) for t in time_scale]
        frames = [sig[t:t + w_len] for t in time_scale]

        pitches = [0.0] * len(time_scale)
        harmonic_rates = [0.0] * len(time_scale)
        argmins = [0.0] * len(time_scale)

        for i, frame in enumerate(frames):
            # Compute YIN
            df = self.difference_function(frame, w_len, tau_max)
            cm_df = self.cumulative_mean_normalized_difference_function(df, tau_max)
            p = self.get_pitch(cm_df, tau_min, tau_max, harmo_thresh)

            # Get results
            if np.argmin(cm_df) > tau_min:
                argmins[i] = float(sr / np.argmin(cm_df))
            if p != 0:  # A pitch was found
                pitches[i] = float(sr / p)
                harmonic_rates[i] = cm_df[p]
            else:  # No pitch, but we compute a value of the harmonic rate
                harmonic_rates[i] = min(cm_df)

        return pitches, harmonic_rates, argmins, times
    
    def difference_function(self, x, n, tau_max):
        """
        Compute difference function of data x. This solution is implemented directly
        with Numpy fft.
        :param x: audio data
        :param n: length of data
        :param tau_max: integration window size
        :return: difference function
        :rtype: list
        """

        x = np.array(x, np.float64)
        w = x.size
        tau_max = min(tau_max, w)
        x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
        size = w + tau_max
        p2 = (size // 32).bit_length()
        nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
        size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
        fc = np.fft.rfft(x, size_pad)
        conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
        return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - \
            2 * conv
        
    
    def cumulative_mean_normalized_difference_function(self, df, n):
        """
        Compute cumulative mean normalized difference function (CMND).
        :param df: Difference function
        :param n: length of data
        :return: cumulative mean normalized difference function
        :rtype: list
        """

        # scipy method
        cmn_df = df[1:] * range(1, n) / np.cumsum(df[1:]).astype(float)
        return np.insert(cmn_df, 0, 1)

    def get_pitch(self, cmdf, tau_min, tau_max, harmo_th=0.1):
        """
        Return fundamental period of a frame based on CMND function.
        :param cmdf: Cumulative Mean Normalized Difference function
        :param tau_min: minimum period for speech
        :param tau_max: maximum period for speech
        :param harmo_th: harmonicity threshold to determine if it is necessary to
        compute pitch frequency
        :return: fundamental period if there is values under threshold, 0 otherwise
        :rtype: float
        """
        tau = tau_min
        while tau < tau_max:
            if cmdf[tau] < harmo_th:
                while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                    tau += 1
                return tau
            tau += 1

        return 0    # if unvoiced


if __name__ == "__main__":
    path = "../Testset/clean/sp01.wav"
    y, sr = torchaudio.load(path)
    pitch = Pitch(sr)
    pitch.plot_f0(y, save_path='../docs/examples/foo.png')