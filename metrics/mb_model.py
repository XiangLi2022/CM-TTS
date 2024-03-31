import torch
import torch.nn as nn

class MBNet(nn.Module):
    def __init__(self, num_judges, activation = nn.ReLU):
        super(MBNet, self).__init__()
        self.mean_net_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            activation(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            activation(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            activation()
        )

        self.mean_net_rnn = nn.LSTM(input_size = 512, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn = nn.Sequential(
            nn.Linear(256, 128),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(128,1),
            activation()
        )
        self.bias_net_first_conv = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride = (1,3))

        self.bias_net_conv = nn.Sequential(
            nn.Conv2d(in_channels = 17, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride = (1,3)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride = (1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            activation()
        )
        self.bias_net_rnn = nn.LSTM(input_size = 128, hidden_size = 64, num_layers = 1, batch_first = True, bidirectional = True)
        self.bias_net_dnn = nn.Sequential(
            nn.Linear(128, 32),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(32,1),
            activation() 
            # Here the activation may not be added. 
            # However, in my testing, I add the activation and get comparable results. 
        )
        self.judge_embedding = nn.Embedding(num_embeddings = num_judges, embedding_dim = 86)

    def get_mean_mos(self, spectrum):
        # print(spectrum.size())
        #spectrum should have shape (batch, 1, time, 257)
        batch = spectrum.shape[0]
        time = spectrum.shape[2]
        mean_feat = self.mean_net_conv(spectrum)
        mean_feat = mean_feat.view((batch, time, 512))
        mean_feat, (h, c) = self.mean_net_rnn(mean_feat)
        return self.mean_net_dnn(mean_feat)  # (batch, seq, 1)

    def forward(self, spectrum, judge_id):
        #spectrum should have shape (batch, 1, time, 257)
        batch = spectrum.shape[0]
        time = spectrum.shape[2]
        mean_feat = self.mean_net_conv(spectrum)
        mean_feat = mean_feat.view((batch, time, 512))
        mean_feat, (h, c) = self.mean_net_rnn(mean_feat)
        mean_feat = self.mean_net_dnn(mean_feat)  # (batch, seq, 1)

        bias_feat = self.bias_net_first_conv(spectrum)
        judge_feat = self.judge_embedding(judge_id) # (batch, feat_dim)
        judge_feat = judge_feat.unsqueeze(1) # (batch, 1, feat_dim)
        judge_feat = torch.stack([judge_feat for i in range(time)], dim = 2) #(batch, 1, time, feat_dim)
        bias_feat = torch.cat([bias_feat, judge_feat], dim = 1)
        bias_feat = self.bias_net_conv(bias_feat)
        bias_feat = bias_feat.view((batch, time, 128))
        bias_feat, (h, c) = self.bias_net_rnn(bias_feat)
        bias_feat = self.bias_net_dnn(bias_feat)
        bias_feat = bias_feat + mean_feat
        return mean_feat, bias_feat
        

    def sample_inference(self, spectrum, judge_id=None):
        bsz = spectrum.shape[0]
        if judge_id==None:
            judge_id = torch.randint(1000, (bsz,4)).to(spectrum.device)
        scores = []
        with torch.no_grad():
            for i in range(4):
                mean_feat, bias_feat = self.forward(spectrum, judge_id[:,i])
                bias_feat = bias_feat.squeeze(-1)
                bias_feat = torch.mean(bias_feat, dim = -1)
                scores.append(bias_feat)
            scores = torch.stack(scores, dim = 1)
            scores = torch.mean(scores, dim = 1)
        return scores
    
    def only_mean_inference(self, spectrum):
        batch = spectrum.shape[0]
        time = spectrum.shape[2]
        mean_feat = self.mean_net_conv(spectrum)
        mean_feat = mean_feat.view((batch, time, 512))
        mean_feat, (h, c) = self.mean_net_rnn(mean_feat)
        mean_feat = self.mean_net_dnn(mean_feat)
        mean_feat = mean_feat.squeeze(-1)
        mean_scores = torch.mean(mean_feat, dim=-1)
        return mean_scores


