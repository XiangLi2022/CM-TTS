import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeaker import embedding

####added by yingting
#'''
from pathlib import Path
from ge2e_encoder import inference as encoder
#'''

class PreDefinedEmbedder(nn.Module):
    """ Speaker Embedder Wrapper """

    def __init__(self, config):
        super(PreDefinedEmbedder, self).__init__()
        self.sampling_rate = config.sampling_rate
        self.win_length = config.win_length
        self.embedder_type = config.speaker_embedder
        self.embedder_cuda = config.speaker_embedder_cuda
        self.embedder = self._get_speaker_embedder()

        self.config = config

    def _get_speaker_embedder(self):
        embedder = None
        if self.embedder_type == "DeepSpeaker":
            embedder = embedding.build_model(
                "./deepspeaker/pretrained_models/ResCNN_triplet_training_checkpoint_265.h5"
            )
        elif self.embedder_type == "GE2E":
            encoder.load_model(Path(self.config.ge2e_speaker_embedder_path))
        else:
            raise NotImplementedError
        return embedder

    def forward(self, audio):
        if self.embedder_type == "DeepSpeaker":
            spker_embed = embedding.predict_embedding(
                self.embedder,
                audio,
                self.sampling_rate,
                self.win_length,
                self.embedder_cuda
            )
        elif self.embedder_type == "GE2E":
            spker_embed = encoder.embed_utterance(audio).reshape(1, -1)

        return spker_embed
