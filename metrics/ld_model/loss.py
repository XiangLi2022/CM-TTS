import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import make_non_pad_mask


class Loss(nn.Module):
    """
    Implements the clipped MSE loss and categorical loss.
    """
    def __init__(self, output_type, alpha, lamb, tau, masked_loss):
        super(Loss, self).__init__()
        self.output_type = output_type
        self.alpha = alpha
        self.lamb = lamb
        self.tau = tau
        self.masked_loss = masked_loss

        if output_type == "scalar":
            criterion = torch.nn.MSELoss
        elif output_type == "categorical":
            criterion = torch.nn.CrossEntropyLoss

        if self.alpha > 0:
            self.mean_net_criterion = criterion(reduction="none")
        self.main_criterion = criterion(reduction="none")

    def forward_criterion(self, y_hat, label, criterion_module, masks=None):
        # might investigate how to combine masked loss with categorical output
        if masks is not None:
            y_hat = y_hat.masked_select(masks)
            label = label.masked_select(masks)

        if self.output_type == "scalar":
            y_hat = y_hat.squeeze(-1)
            loss = criterion_module(y_hat, label)
            threshold = torch.abs(y_hat - label) > self.tau
            loss = torch.mean(threshold * loss)
        elif self.output_type == "categorical":
            # y_hat must have shape [..., 5]
            y_hat = y_hat.view((-1, 5))
            label = torch.flatten(label).type(torch.long)
            loss = criterion_module(y_hat, label-1)
            loss = torch.mean(loss)
        return loss

    def forward(self, pred_mean, gt_mean, pred_score, gt_score, lens, device):
        """
        Args:
            pred_mean, pred_score: [batch, time, 1/5]
        """
        # make mask
        if self.masked_loss:
            masks = make_non_pad_mask(lens).to(device)
        else:
            masks = None
        
        # repeat for frame level loss
        time = pred_score.shape[1]
        gt_mean = gt_mean.unsqueeze(1).repeat(1, time)
        gt_score = gt_score.unsqueeze(1).repeat(1, time)

        main_loss = self.forward_criterion(pred_score, gt_score, self.main_criterion, masks)
        if self.alpha > 0:
            mean_net_loss = self.forward_criterion(pred_mean, gt_mean, self.mean_net_criterion, masks)
            return self.alpha * mean_net_loss + self.lamb * main_loss, mean_net_loss, main_loss
        else:
            return self.lamb * main_loss, None, main_loss

#####################################################################################

# Categorical loss was not useful in initial experiments, but I keep it here for future reference

class CategoricalLoss(nn.Module):
    def __init__(self, alpha, lamb):
        super(CategoricalLoss, self).__init__()
        self.alpha = alpha
        self.lamb = lamb

        if self.alpha > 0:
            self.mean_net_criterion = nn.CrossEntropyLoss()
        self.main_criterion = nn.CrossEntropyLoss()
    
    def ce(self, y_hat, label, criterion, masks=None):
        if masks is not None:
            y_hat = y_hat.masked_select(masks)
            label = label.masked_select(masks)
        ce = criterion(y_hat, label-1)
        return ce

    def forward(self, pred_mean, gt_mean, pred_score, gt_score, lens, device):
        # make mask
        if self.masked_loss:
            masks = make_non_pad_mask(lens).to(device)
        else:
            masks = None
            
        score_ce = self.ce(pred_score, gt_score, self.main_criterion, masks)
        if self.alpha > 0:
            mean_ce = self.ce(pred_mean, gt_mean, self.mean_net_criterion, masks)
            return self.alpha * mean_ce + self.lamb * score_ce, mean_ce, score_ce
        else:
            return self.lamb * score_ce, None, score_ce
