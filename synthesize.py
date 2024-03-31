import os
import re
import json
import argparse
from string import punctuation
from tqdm import tqdm

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples
from dataset import Dataset, TextDataset
from text import text_to_sequence
from model.cm_tool import dist_util, logger
from model.cm_tool.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_model_and_diffusion_tts,
    add_dict_to_argparser,
    args_to_dict,
)
import os.path as osp
from model.cm_tool.random_util import get_generator
from model.cm_tool.karras_diffusion import karras_sample, karras_sample_tts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CMTotalTTSSynthesize:
    def __init__(self,
                 model_path, model_step_num,
                 args, preprocess_config, model_config, train_config,
                 p_control=1.0,
                 e_control=1.0,
                 d_control=1.0,
                 ):
        self.device = dist_util.dev()
        self.CMDenoiserTTS_path = osp.join(
            model_path,
            "CMDenoiserTTS",
            "model{:06d}.pt".format(model_step_num)
        )
        self.model, self.diffusion = self.load_cm_model(args, preprocess_config, model_config, train_config)

        self.duration_pitch_energy_net, self.denoise_net = self.model.get_segmentation_model()

        self.p_control = p_control
        self.e_control = e_control
        self.d_control = d_control

        self.train_config = train_config

    def load_cm_model(self, args, preprocess_config, model_config, train_config):
        args_cm = argparse.Namespace(**train_config["cm"])
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
            "args": args,
            "train_config": train_config,
            "preprocess_config": preprocess_config,
            "model_config": model_config,
        }
        model, diffusion = create_model_and_diffusion_tts(**model_and_diffusion_kwargs)
        model.load_state_dict(
            torch.load(
                self.CMDenoiserTTS_path, map_location=dist_util.dev()
            ),
        )
        model.to(self.device)
        model.eval()
        return model, diffusion

    def synthesize(self, batch):
        args_cm = argparse.Namespace(**self.train_config["cm"])
        """
        speakers,  # 2
                texts,  # 3
                src_lens,  # 4
                spker_embeds
        """
        duration_pitch_dict = {
            "speakers": batch[2],
            "texts": batch[3],
            "src_lens": batch[4],
            "spker_embeds": batch[-1]
        }
        out_dict = self.duration_pitch_energy_net(**duration_pitch_dict)
        cond_dict = dict(
            conditioner=out_dict["cond"],
            speaker_emb=out_dict["speaker_emb"],
            mask=out_dict["mel_masks"],
        )

        batch_size, seq_len, _ = cond_dict["conditioner"].size()

        if args.T == 1:
            sample = karras_sample_tts(
                diffusion=self.diffusion,
                model=self.model,
                shape=(batch_size, 1, seq_len, 80),
                model_kwargs=duration_pitch_dict,
                device=self.device,
                sigma_max=args_cm.sigma_max,
                sigma_min=args_cm.sigma_min,
                sampler="onestep",
            )
        elif args.T == 2:
            sample = karras_sample_tts(
                diffusion=self.diffusion,
                model=self.model,
                shape=(batch_size, 1, seq_len, 80),
                model_kwargs=duration_pitch_dict,
                device=self.device,
                sigma_max=args_cm.sigma_max,
                sigma_min=args_cm.sigma_min,
                sampler="multistep",
                steps=2,
                ts=(0,0,1),
            )
        elif args.T == 4:
            sample = karras_sample_tts(
                diffusion=self.diffusion,
                model=self.model,
                shape=(batch_size, 1, seq_len, 80),
                model_kwargs=duration_pitch_dict,
                device=self.device,
                sigma_max=args_cm.sigma_max,
                sigma_min=args_cm.sigma_min,
                sampler="multistep",
                steps=2,
                ts=(0,0,0,0,1),
            )
        out_put = [None] * 12
        out_put[0] = sample
        out_put[10] = duration_pitch_dict["src_lens"]  # 10
        out_put[11] = out_dict["mel_lens"]

        return out_put  # (batch_size, seq_len, 80)


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize_cm(model_path, model_step_num, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    def synthesize_(batch):
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            cm_synthesize_tool = CMTotalTTSSynthesize(
                model_path, model_step_num, args, preprocess_config, model_config, train_config,
                p_control=pitch_control, e_control=energy_control, d_control=duration_control
            )
            out_put = cm_synthesize_tool.synthesize(batch)
            synth_samples(
                args,
                batch,
                out_put,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                cm_synthesize_tool.diffusion,  # 这个玩意应该没啥作用
            )

    if args.teacher_forced:
        for batchs_ in batchs:
            for batch in tqdm(batchs_):
                batch = list(batch)
                # batch[6] = None  # set mel None for diffusion sampling
                synthesize_(batch)
    else:
        for batch in tqdm(batchs):
            synthesize_(batch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=False,
        default="naive",
        help="training model type",
    )
    parser.add_argument("--teacher_forced", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="p225",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=str,
        default=1.0,
        help="指定模型位置",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="指定数据输出位置",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.text is None
        if args.teacher_forced:
            assert args.source is None
        else:
            assert args.source is not None
    if args.mode == "single":
        assert args.source is None and args.text is not None and not args.teacher_forced

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)

    train_tag = "naive"

    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"] + "_{}{}".format(train_tag, path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"] + "_{}{}".format(train_tag, path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"] + "_{}{}".format(args.model, path_tag)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt

        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]
    if args.result_path is not None:
        train_config["path"]["result_path"] = args.result_path

    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Log Configuration
    print("\n==================================== Inference Configuration ====================================")
    print(" ---> Type of Modeling:", args.model)
    print(" ---> Total Batch Size:", int(train_config["optimizer"]["batch_size"]))
    print(" ---> Path of ckpt:", train_config["path"]["ckpt_path"])
    print(" ---> Path of log:", train_config["path"]["log_path"])
    print(" ---> Path of result:", train_config["path"]["result_path"])
    print("================================================================================================")

    # Get model
    # model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    # args.teacher_forced=True

    if args.mode == "batch":
        # Get dataset
        if args.teacher_forced:
            dataset = Dataset(
                "val.txt", args, preprocess_config, model_config, train_config, sort=False, drop_last=False
            )
        else:
            dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    # Preprocess texts
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
                           and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array(
            [0])  # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            raise NotImplementedError
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize_cm(
        model_path=args.model_path, model_step_num=args.restore_step,
        args=args, configs=configs, vocoder=vocoder, batchs=batchs, control_values=control_values)
