# from model.cm_tool.train_util import CMTrainLoop
import argparse
import copy
import json

import numpy as np
import torch

import hifigan
from model.cm_tool import dist_util, logger
# from ..model.cm_tool.image_datasets import load_data
from model.cm_tool.resample import create_named_schedule_sampler_num_scales
from model.cm_tool.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion_tts,
    args_to_dict,
    create_ema_and_scales_fn,
)
from model.cm_tool.train_util import CMTTSTrainTool


def get_model_cm(args, preprocess_config, model_config, train_config,train=True):
    """
    cm model init
    :return:
    """

    args_cm = argparse.Namespace(**train_config["cm"])

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args_cm.target_ema_mode,
        start_ema=args_cm.start_ema,
        scale_mode=args_cm.scale_mode,
        start_scales=args_cm.start_scales,
        end_scales=args_cm.end_scales,
        total_steps=args_cm.total_training_steps,
        distill_steps_per_iter=args_cm.distill_steps_per_iter,
    )
    if args_cm.training_mode == "progdist":
        distillation = False
    elif "consistency" in args_cm.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args_cm.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args_cm, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model_and_diffusion_kwargs["tts_model_config"] = {
        # create model
        "args": args,
        "train_config":train_config,
        "preprocess_config": preprocess_config,
        "model_config": model_config,
    }
    model, diffusion = create_model_and_diffusion_tts(**model_and_diffusion_kwargs)

    model.to(dist_util.dev())
    model.train()
    if args_cm.use_fp16:
        model.convert_to_fp16()
    # only for double fixed EMA
    schedule_sampler = create_named_schedule_sampler_num_scales(args_cm.schedule_sampler, args_cm.start_scales)
    # can be None
    if args_cm.teacher_model_path is not None and len(args_cm.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args_cm.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args_cm.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion_tts(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(args_cm.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args_cm.use_fp16:
            teacher_model.convert_to_fp16()
    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.
    logger.log("creating the target model")
    target_model, _ = create_model_and_diffusion_tts(
        **model_and_diffusion_kwargs,
    )
    target_model.to(dist_util.dev())
    target_model.train()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    print("params_num:", get_param_num(model))  # 

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args_cm.use_fp16:
        target_model.convert_to_fp16()

    logger.log("training...")
    train_tool = CMTTSTrainTool(
        model=model,
        diffusion=diffusion,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args_cm.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args_cm.total_training_steps,
        # data=data,
        batch_size=train_config["optimizer"]["batch_size"],
        microbatch=args_cm.microbatch,
        lr=args_cm.lr,
        ema_rate=args_cm.ema_rate,
        log_interval=args_cm.log_interval,
        save_interval=args_cm.save_interval,
        resume_checkpoint=args_cm.resume_checkpoint,
        use_fp16=args_cm.use_fp16,
        fp16_scale_growth=args_cm.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args_cm.weight_decay,
        lr_anneal_steps=args_cm.lr_anneal_steps,
    )
    return train_tool

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_netG_params(model_kernel):
    return list(model_kernel.C.parameters()) \
        + list(model_kernel.Z.parameters()) \
        + list(model_kernel.G.parameters())


def get_netD_params(model_kernel):
    return model_kernel.D.parameters()


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
