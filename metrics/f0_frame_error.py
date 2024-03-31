import numpy as np
import torch
import torchaudio

from .pitchTracking import Pitch

class FFE:
    def __init__(self, sr=16000):
        self.sr = sr
        self.pitch = Pitch(self.sr)

    def __call__(self, y_ref, y_syn):
        return self.calculate_ffe(y_ref, y_syn)

    def calculate_ffe_path(self, ref_path, syn_path):
        y_ref, sr_ref = torchaudio.load(ref_path)
        y_syn, sr_syn = torchaudio.load(syn_path)
        assert sr_ref == sr_syn, f"{sr_ref} != {sr_syn}" # audios of same sr
        assert sr_ref == self.sr, f"{sr_ref} != {self.sr}" # sr of audio and pitch tracking is same
        return self.calculate_ffe(y_ref, y_syn)

    def calculate_ffe(self, y_ref, y_syn):
        y_ref = y_ref.view(-1)
        y_syn = y_syn.view(-1)
        yref_f, _, _, yref_t = self.pitch.compute_yin(y_ref, self.sr)
        ysyn_f, _, _, ysyn_t = self.pitch.compute_yin(y_syn, self.sr)

        yref_f = np.array(yref_f)
        yref_t = np.array(yref_t)
        ysyn_f = np.array(ysyn_f)
        ysyn_t = np.array(ysyn_t)

        distortion = self.f0_frame_error(yref_t, yref_f, ysyn_t, ysyn_f)
        return distortion.item()


    def f0_frame_error(self, true_t, true_f, est_t, est_f):
        gross_pitch_error_frames = self._gross_pitch_error_frames(
            true_t, true_f, est_t, est_f
        )
        voicing_decision_error_frames = self._voicing_decision_error_frames(
            true_t, true_f, est_t, est_f
        )
        return (np.sum(gross_pitch_error_frames) +
                np.sum(voicing_decision_error_frames)) / (len(true_t))

    def _voicing_decision_error_frames(self, true_t, true_f, est_t, est_f):
        return (est_f != 0) != (true_f != 0)

    def _true_voiced_frames(self, true_t, true_f, est_t, est_f):
        return (est_f != 0) & (true_f != 0)

    def _gross_pitch_error_frames(self, true_t, true_f, est_t, est_f, eps=1e-8):
        voiced_frames = self._true_voiced_frames(true_t, true_f, est_t, est_f)
        true_f_p_eps = [x + eps for x in true_f]
        pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
        return voiced_frames & pitch_error_frames

if __name__ == "__main__":
    path1 = "../docs/audio/sp15.wav"
    path2 = "../docs/noisy/sp15_station_sn5.wav"

    ffe = FFE(22050)
    print(ffe.calculate_ffe_path(path1, path2))

    y_ref, sr_ref = torchaudio.load(path1)
    y_syn, sr_syn = torchaudio.load(path2)

    print(y_ref.size(), y_syn.size())
    print(ffe.calculate_ffe(y_ref, y_syn))
