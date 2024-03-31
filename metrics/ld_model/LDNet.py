import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .modules import Projection, MobileNetV2ConvBlocks, MobileNetV3ConvBlocks, STRIDE


class LDNet(nn.Module):
    def __init__(self, config):
        super(LDNet, self).__init__()
        self.config = config

        # This is not really used but we just keep it
        if config["combine_mean_score"]:
            assert config["output_type"] == "scalar"
            assert config["use_mean_net"]
            assert config["mean_net_output_type"] == config["output_type"]

        # define judge embedding
        self.num_judges = config["num_judges"]
        self.judge_embedding = nn.Embedding(num_embeddings = self.num_judges, embedding_dim = config["judge_emb_dim"])

        # define activation
        if config["activation"] == "ReLU":
            activation = nn.ReLU
        else:
            raise NotImplementedError

        # define encoder
        if config["encoder_type"] == "mobilenetv2":
            self.encoder = MobileNetV2ConvBlocks(config["encoder_conv_first_ch"],
                                                 config["encoder_conv_t"],
                                                 config["encoder_conv_c"],
                                                 config["encoder_conv_n"],
                                                 config["encoder_conv_s"],
                                                 config["encoder_output_dim"])
        elif config["encoder_type"] == "mobilenetv3":
            self.encoder = MobileNetV3ConvBlocks(config["encoder_bneck_configs"],
                                                 config["encoder_output_dim"])
        else:
            raise NotImplementedError

        # define decoder
        if config["decoder_type"] == "ffn":
            decoder_dnn_input_dim = config["encoder_output_dim"] + config["judge_emb_dim"]
        elif config["decoder_type"] == "rnn":
            self.decoder_rnn = nn.LSTM(input_size = config["encoder_output_dim"] + config["judge_emb_dim"],
                                       hidden_size = config["decoder_rnn_dim"],
                                       num_layers = 1, batch_first = True, bidirectional = True)
            decoder_dnn_input_dim = config["decoder_rnn_dim"] * 2
        # there is always dnn
        self.decoder_dnn = Projection(decoder_dnn_input_dim, config["decoder_dnn_dim"],
                                      activation, config["output_type"], config["range_clipping"])
        
        # define mean net
        if config["use_mean_net"]:
            if config["mean_net_type"] == "ffn":
                mean_net_dnn_input_dim = config["encoder_output_dim"]
            elif config["mean_net_type"] == "rnn":
                self.mean_net_rnn = nn.LSTM(input_size = config["encoder_output_dim"],
                                           hidden_size = config["mean_net_rnn_dim"],
                                           num_layers = 1, batch_first = True, bidirectional = True)
                mean_net_dnn_input_dim = config["mean_net_rnn_dim"] * 2
            # there is always dnn
            self.mean_net_dnn = Projection(mean_net_dnn_input_dim, config["mean_net_dnn_dim"],
                                          activation, config["output_type"], config["mean_net_range_clipping"])

    def _get_output_dim(self, input_size, num_layers, stride=STRIDE):
        """
        calculate the final ouptut width (dim) of a CNN using the following formula
        w_i = |_ (w_i-1 - 1) / stride + 1 _|
        """
        output_dim = input_size
        for _ in range(num_layers):
            output_dim = math.floor((output_dim-1)/STRIDE+1)
        return output_dim

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())

    def forward(self, spectrum, judge_id):
        """Calculate forward propagation.
            Args:
                spectrum has shape (batch, time, dim)
                judge_id has shape (batch)
        """
        batch, time, dim = spectrum.shape
        
        # get judge embedding
        judge_feat = self.judge_embedding(judge_id) # (batch, emb_dim)
        judge_feat = torch.stack([judge_feat for i in range(time)], dim = 1) #(batch, time, feat_dim)
        
        # encoder and inject judge embedding
        if self.config["encoder_type"] in ["mbnetstyle", "mobilenetv2", "mobilenetv3"]:
            spectrum = spectrum.unsqueeze(1)
            encoder_outputs = self.encoder(spectrum) # (batch, ch, time, feat_dim)
            encoder_outputs = encoder_outputs.view((batch, time, -1)) # (batch, time, feat_dim)
            decoder_inputs = torch.cat([encoder_outputs, judge_feat], dim = -1) # concat along feature dimension
        else:
            raise NotImplementedError
        
        # mean net
        if self.config["use_mean_net"]:
            mean_net_inputs = encoder_outputs
            if self.config["mean_net_type"] == "rnn":
                mean_net_outputs, (h, c) = self.mean_net_rnn(mean_net_inputs)
            else:
                mean_net_outputs = mean_net_inputs
            mean_net_outputs = self.mean_net_dnn(mean_net_outputs) # [batch, time, 1 (scalar) / 5 (categorical)]

        # decoder
        if self.config["decoder_type"] == "rnn":
            decoder_outputs, (h, c) = self.decoder_rnn(decoder_inputs)
        else:
            decoder_outputs = decoder_inputs
        decoder_outputs = self.decoder_dnn(decoder_outputs) # [batch, time, 1 (scalar) / 5 (categorical)]

        # define scores
        mean_score = mean_net_outputs if self.config["use_mean_net"] else None
        ld_score = decoder_outputs

        return mean_score, ld_score

    def mean_listener_inference(self, spectrum):
        assert self.config["use_mean_listener"]
        batch, time, dim = spectrum.shape
        device = spectrum.device
        
        # get judge embedding
        judge_id = (torch.ones(batch, dtype=torch.long) * self.num_judges - 1).to(device) # (bs)
        judge_feat = self.judge_embedding(judge_id) # (bs, emb_dim)
        judge_feat = torch.stack([judge_feat for i in range(time)], dim = 1) #(batch, time, feat_dim)
        
        # encoder and inject judge embedding
        if self.config["encoder_type"] in ["mobilenetv2", "mobilenetv3"]:
            spectrum = spectrum.unsqueeze(1)
            encoder_outputs = self.encoder(spectrum) # (batch, ch, time, feat_dim)
            encoder_outputs = encoder_outputs.view((batch, time, -1)) # (batch, time, feat_dim)
            decoder_inputs = torch.cat([encoder_outputs, judge_feat], dim = -1) # concat along feature dimension
        else:
            raise NotImplementedError

        # decoder
        if self.config["decoder_type"] == "rnn":
            decoder_outputs, (h, c) = self.decoder_rnn(decoder_inputs)
        else:
            decoder_outputs = decoder_inputs
        decoder_outputs = self.decoder_dnn(decoder_outputs) # [batch, time, 1 (scalar) / 5 (categorical)]

        # define scores
        decoder_outputs = decoder_outputs.squeeze(-1)
        scores = torch.mean(decoder_outputs, dim=1)
        return scores

    def average_inference(self, spectrum, include_meanspk=False):
        bs, time, _ = spectrum.shape
        device = spectrum.device
        if self.config["use_mean_listener"] and not include_meanspk:
            actual_num_judges = self.num_judges - 1
        else:
            actual_num_judges = self.num_judges
        
        # all judge ids
        judge_id = torch.arange(actual_num_judges, dtype=torch.long).repeat(bs, 1).to(device) # (bs, nj)
        judge_feat = self.judge_embedding(judge_id) # (bs, nj, emb_dim)
        judge_feat = torch.stack([judge_feat for i in range(time)], dim = 2) # (bs, nj, time, feat_dim)
        
        # encoder and inject judge embedding
        if self.config["encoder_type"] in ["mobilenetv2", "mobilenetv3"]:
            spectrum = spectrum.unsqueeze(1)
            encoder_outputs = self.encoder(spectrum) # (batch, ch, time, feat_dim)
            encoder_outputs = encoder_outputs.view((bs, time, -1)) # (batch, time, feat_dim)
            decoder_inputs = torch.stack([encoder_outputs for i in range(actual_num_judges)], dim = 1) # (bs, nj, time, feat_dim)
            decoder_inputs = torch.cat([decoder_inputs, judge_feat], dim = -1) # concat along feature dimension
        else:
            raise NotImplementedError
        
        # mean net
        if self.config["use_mean_net"]:
            mean_net_inputs = encoder_outputs
            if self.config["mean_net_type"] == "rnn":
                mean_net_outputs, (h, c) = self.mean_net_rnn(mean_net_inputs)
            else:
                mean_net_outputs = mean_net_inputs
            mean_net_outputs = self.mean_net_dnn(mean_net_outputs) # [batch, time, 1 (scalar) / 5 (categorical)]
        
        # decoder
        if self.config["decoder_type"] == "rnn":
            decoder_outputs = decoder_inputs.view((bs * actual_num_judges, time, -1))
            decoder_outputs, (h, c) = self.decoder_rnn(decoder_outputs)
        else:
            decoder_outputs = decoder_inputs
        decoder_outputs = self.decoder_dnn(decoder_outputs) # [batch, time, 1 (scalar) / 5 (categorical)]
        decoder_outputs = decoder_outputs.view((bs, actual_num_judges, time, -1)) # (bs, nj, time, 1/5)

        if self.config["output_type"] == "scalar":
            decoder_outputs = decoder_outputs.squeeze(-1) # (bs, nj, time)
            posterior_scores = torch.mean(decoder_outputs, dim=2)
            ld_scores = torch.mean(decoder_outputs, dim=1) # (bs, time)
        elif self.config["output_type"] == "categorical":
            ld_posterior = torch.nn.functional.softmax(decoder_outputs, dim=-1)
            ld_scores = torch.inner(ld_posterior, torch.Tensor([1,2,3,4,5]).to(device))
            posterior_scores = torch.mean(ld_scores, dim=2)
            ld_scores = torch.mean(ld_scores, dim=1) # (bs, time)

        # define scores
        scores = torch.mean(ld_scores, dim = 1)
        return scores, posterior_scores