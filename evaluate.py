import argparse
import os
import json

import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tools import to_device, log, synth_one_sample,synth_one_sample_diff,log_diff
from model import DiffGANTTSLoss,DiffSingerLoss, CMLoss
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args, model, discriminator, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = DiffGANTTSLoss(args, preprocess_config, model_config, train_config).to(device)

    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)

            with torch.no_grad():
                if args.model == "aux":

                    # Forward
                    output, p_targets, coarse_mels = model(*(batch[2:]))
                    # Update Batch
                    batch[9] = p_targets

                    (
                        fm_loss,
                        recon_loss,
                        mel_loss,
                        pitch_loss,
                        energy_loss,
                        duration_loss,
                    ) = Loss(
                        model,
                        batch,
                        output,
                    )
                    output[0] = output[0][0] # only x_0 is needed after calculating loss

                    G_loss = recon_loss
                    D_loss = fm_loss = adv_loss = torch.zeros(1).to(device)

                else: # args.model in ["naive", "shallow"]

                    #######################
                    # Evaluate Discriminator #
                    #######################

                    # Forward
                    output, *_ = model(*(batch[2:]))

                    xs, spk_emb, t, mel_masks = *(output[1:4]), output[9]
                    x_ts, x_t_prevs, x_t_prev_preds, spk_emb, t = \
                        [x.detach() if x is not None else x for x in (list(xs) + [spk_emb, t])]

                    D_real_cond, D_real_uncond = discriminator(x_ts, x_t_prevs, spk_emb, t)
                    D_fake_cond, D_fake_uncond = discriminator(x_ts, x_t_prev_preds, spk_emb, t)

                    D_loss_real, D_loss_fake = Loss.d_loss_fn(D_real_cond[-1], D_real_uncond[-1], D_fake_cond[-1], D_fake_uncond[-1])

                    D_loss = D_loss_real + D_loss_fake

                    #######################
                    # Evaluate Generator #
                    #######################

                    # Forward
                    output, p_targets, coarse_mels = model(*(batch[2:]))
                    # Update Batch
                    batch[9] = p_targets

                    (x_ts, x_t_prevs, x_t_prev_preds), spk_emb, t, mel_masks = *(output[1:4]), output[9]

                    D_fake_cond, D_fake_uncond = discriminator(x_ts, x_t_prev_preds, spk_emb, t)
                    D_real_cond, D_real_uncond = discriminator(x_ts, x_t_prevs, spk_emb, t)

                    adv_loss = Loss.g_loss_fn(D_fake_cond[-1], D_fake_uncond[-1])

                    (
                        fm_loss,
                        recon_loss,
                        mel_loss,
                        pitch_loss,
                        energy_loss,
                        duration_loss,
                    ) = Loss(
                        model,
                        batch,
                        output,
                        coarse_mels,
                        (D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond),
                    )

                    G_loss = recon_loss + fm_loss + adv_loss

                losses = [D_loss + G_loss, D_loss, G_loss, recon_loss, fm_loss, adv_loss, mel_loss, pitch_loss, energy_loss, duration_loss]

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_msg = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_msg.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_msg.append(loss_sum / len(dataset))
    loss_means_msg = loss_means_msg[0:2] + loss_means_msg[5:]

    message = "Validation Step {}, Total Loss: {:.4f}, D_loss: {:.4f}, adv_loss: {:.4f}, mel_loss: {:.4f}, pitch_loss: {:.4f}, energy_loss: {:.4f}, duration_loss: {:.4f}".format(
        *([step] + [l for l in loss_means_msg])
    )

    if logger is not None:
        figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            args,
            batch,
            output,
            coarse_mels,
            vocoder,
            model_config,
            preprocess_config,
            model.module.diffusion,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            step,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/reconstructed",
        )
        log(
            logger,
            step,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/synthesized",
        )

    return message


def evaluate_diff(args, model, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs
    # Get dataset
    dataset = Dataset(
        "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = DiffSingerLoss(args, preprocess_config, model_config, train_config).to(device)

    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output, p_targets = model(*(batch[2:]))
                # Update Batch
                batch[9] = p_targets

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Noise Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means_])
    )

    if logger is not None:
        figs, wav_reconstruction, wav_prediction, tag = synth_one_sample_diff(
            args,
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
            model.module.diffusion,
        )

        log_diff(logger, step, losses=loss_means)
        log_diff(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log_diff(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log_diff(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message

def evaluate_cm(args, duration_pith_speaker_model, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = CMLoss(args, preprocess_config, model_config, train_config).to(device)

    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                out_dict = duration_pith_speaker_model(*(batch[2:]))

                cond_dict = {
                    "conditioner": out_dict["cond"],
                    "speaker_emb": out_dict["speaker_emb"],
                    "mask": out_dict["mel_masks"],
                }
                mel_predictions = None  # 这里需要填充预测
                loss_calculate_dict = {
                    "pitch_predictions": out_dict["p_predictions"],
                    "energy_predictions": out_dict["e_predictions"],
                    "log_duration_predictions": out_dict["log_d_predictions"],
                    "src_masks": out_dict["src_masks"],
                    "mel_masks": out_dict["mel_masks"],
                    "mel_predictions": mel_predictions,
                }

                # Cal Loss
                losses = Loss(batch, loss_calculate_dict)

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Noise Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means_])
    )
    """
    if logger is not None:
        figs, wav_reconstruction, wav_prediction, tag = synth_one_sample_diff(
            args,
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
            duration_pith_speaker_model.module.diffusion,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )
    """
    return message