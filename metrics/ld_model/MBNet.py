import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Projection, MBNetConvBlocks


class MBNet(nn.Module):
    def __init__(self, config):
        super(MBNet, self).__init__()
        self.config = config

        # sanity check for MBNet
        assert config["combine_mean_score"]
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

        # define mean net
        if config["use_mean_net"]:
            if config["mean_net_input"] == "audio":
                self.mean_net_conv = MBNetConvBlocks(1,
                                                config["mean_net_conv_chs"],
                                                config["mean_net_dropout_rate"],
                                                activation
                                                )
            else:
                raise NotImplementedError

            self.mean_net_rnn = nn.LSTM(input_size = config["mean_net_conv_chs"][-1] * 4,
                                        hidden_size = config["mean_net_rnn_dim"],
                                        num_layers = 1, batch_first = True, bidirectional = True)
            self.mean_net_dnn = Projection(config["mean_net_rnn_dim"] * 2, config["mean_net_dnn_dim"],
                                           activation, config["mean_net_output_type"], config["mean_net_range_clipping"])

        # define encoder and decoder (a.k.a. bias net)
        self.encoder = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride = (1,3))
        self.decoder_conv = MBNetConvBlocks(17,
                                           config["decoder_conv_chs"],
                                           config["decoder_dropout_rate"],
                                           activation
                                           )
        decoder_conv_output_dim = self._get_decoder_conv_output_dim(config["audio_input_dim"], len(config["decoder_conv_chs"])+1) # +1: encoder conv
        self.decoder_rnn = nn.LSTM(input_size = config["decoder_conv_chs"][-1] * decoder_conv_output_dim,
                                   hidden_size = config["decoder_rnn_dim"],
                                   num_layers = 1, batch_first = True, bidirectional = True)
        self.decoder_dnn = Projection(config["decoder_rnn_dim"] * 2, config["decoder_dnn_dim"],
                                      activation, config["output_type"], config["range_clipping"])

    def _get_decoder_conv_output_dim(self, decoder_conv_input_size, decoder_conv_num_layers):
        # w_out = |_ (w_in - 1) / 3 + 1 _|
        output_dim = decoder_conv_input_size
        for _ in range(decoder_conv_num_layers):
            output_dim = math.floor((output_dim-1)/3+1)
        return output_dim

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())

    def forward(self, spectrum, judge_id):
        """Calculate forward propagation.
            Args:
                spectrum has shape (batch, time, dim)
                judge_id has shape (batch)
        """
        batch, time, _ = spectrum.shape
        
        # get judge embedding
        judge_feat = self.judge_embedding(judge_id) # (batch, emb_dim)
        judge_feat = torch.stack([judge_feat for i in range(time)], dim = 1) #(batch, time, feat_dim)
        
        # encoder and inject judge embedding
        spectrum = spectrum.unsqueeze(1)
        encoder_outputs = self.encoder(spectrum)
        judge_feat = judge_feat.unsqueeze(1) # (batch, 1, time, feat_dim)
        encoder_outputs = torch.cat([encoder_outputs, judge_feat], dim = 1) # concat along channel dimension, resulting in shape [batch, ch, t, d]

        # decoder
        decoder_outputs = self.decoder_conv(encoder_outputs)
        decoder_outputs = decoder_outputs.view((batch, time, -1))
        decoder_outputs, _ = self.decoder_rnn(decoder_outputs)
        decoder_outputs = self.decoder_dnn(decoder_outputs) # [batch, time, 1 (scalar) / 5 (categorical)]
        
        # mean net
        if self.config["use_mean_net"]:
            if self.config["mean_net_input"] == "audio":
                mean_net_inputs = spectrum
                #mean_net_inputs = mean_net_inputs.unsqueeze(1)
            else:
                raise NotImplementedError
            mean_net_outputs = self.mean_net_conv(mean_net_inputs)
            mean_net_outputs = mean_net_outputs.view((batch, time, 512))
            mean_net_outputs, _ = self.mean_net_rnn(mean_net_outputs)
            mean_net_outputs = self.mean_net_dnn(mean_net_outputs) # (batch, time, 1 (scalar) / 5 (categorical)

        # define scores
        mean_score = mean_net_outputs if self.config["use_mean_net"] else None
        ld_score = decoder_outputs + mean_net_outputs if self.config["combine_mean_score"] else decoder_outputs

        return mean_score, ld_score
    
    def only_mean_inference(self, spectrum):
        assert self.config["use_mean_net"]
        batch, time, _ = spectrum.shape
        spectrum = spectrum.unsqueeze(1)
            
        if self.config["mean_net_input"] == "audio":
            mean_net_inputs = spectrum
        else:
            raise NotImplementedError
        mean_net_outputs = self.mean_net_conv(mean_net_inputs)
        mean_net_outputs = mean_net_outputs.view((batch, time, 512))
        mean_net_outputs, _ = self.mean_net_rnn(mean_net_outputs)
        mean_net_outputs = self.mean_net_dnn(mean_net_outputs) # (batch, seq, 1)
        mean_net_outputs = mean_net_outputs.squeeze(-1)

        mean_scores = torch.mean(mean_net_outputs, dim = -1)
        return mean_scores
    
    def average_inference_v1(self, spectrum, include_meanspk=False):
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
        if self.config["encoder_type"] == "simple":
            # encoder_outputs has shape [batch, ch, t, d]
            spectrum = spectrum.unsqueeze(1)
            encoder_outputs = self.encoder(spectrum)
            encoder_outputs = torch.stack([encoder_outputs for i in range(actual_num_judges)], dim = 1) # (bs, nj, ch, time, feat_dim)
            judge_feat = judge_feat.unsqueeze(2) # (batch, nj, 1, time, feat_dim)
            encoder_outputs = torch.cat([encoder_outputs, judge_feat], dim = 2) # concat along channel dimension
        elif self.config["encoder_type"] == "taco2":
            # encoder_outputs has shape [batch, t, d]
            # concat along feature dimension
            encoder_outputs = self.encoder(spectrum) # (bs, time, hidden_dim)
            encoder_outputs = torch.stack([encoder_outputs for i in range(actual_num_judges)], dim = 1) # (bs, nj, time, feat_dim)
            encoder_outputs = torch.cat([encoder_outputs, judge_feat], dim = 3)
            encoder_outputs = encoder_outputs.unsqueeze(2) # (batch, nj, 1, time, feat_dim)
        else:
            raise NotImplementedError
        
        # decoder conv
        encoder_outputs = encoder_outputs.view(-1, *encoder_outputs.shape[-3:])
        decoder_outputs = self.decoder_conv(encoder_outputs)

        # decoder rnn
        decoder_outputs = decoder_outputs.view((bs * actual_num_judges, time, -1))
        decoder_outputs, (h, c) = self.decoder_rnn(decoder_outputs)
        decoder_outputs = self.decoder_dnn(decoder_outputs) # (bs * nj, time, 1/5)
        decoder_outputs = decoder_outputs.view((bs, actual_num_judges, time, -1)) # (bs, nj, time, 1/5)
        if self.config["output_type"] == "scalar":
            decoder_outputs = decoder_outputs.squeeze(-1)
            posterior_scores = torch.mean(decoder_outputs, dim=2)
            ld_scores = torch.mean(decoder_outputs, dim=1) # (bs, time)
        elif self.config["output_type"] == "categorical":
            ld_posterior = torch.nn.functional.softmax(decoder_outputs, dim=-1)
            ld_scores = torch.inner(ld_posterior, torch.Tensor([1,2,3,4,5]).to(device))
            ld_scores = torch.mean(ld_scores, dim=1) # (bs, time)
        
        # mean net
        if self.config["use_mean_net"]:
            if self.config["mean_net_input"] == "audio":
                mean_net_inputs = spectrum
            else:
                raise NotImplementedError
            #print("mean net input shape", mean_net_inputs.shape)
            mean_net_outputs = self.mean_net_conv(mean_net_inputs)
            mean_net_outputs = mean_net_outputs.view((bs, time, -1))
            mean_net_outputs, (h, c) = self.mean_net_rnn(mean_net_outputs)
            mean_net_outputs = self.mean_net_dnn(mean_net_outputs) # (batch, seq, 1)
            mean_scores = mean_net_outputs.squeeze(-1) # (bs, time)

        # define scores
        if self.config["combine_mean_score"]:
            ld_scores += mean_scores
        scores = torch.mean(ld_scores, dim = 1)
        return scores, posterior_scores