import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate_cm
from model.loss import MelLoss
from utils.model import get_vocoder, get_param_num, get_model_cm
from utils.tools import get_configs_of, to_device, log_cm
from utils.tools import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", args, preprocess_config, model_config, train_config, sort=True, drop_last=True
    )

    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    cm_train_tool = get_model_cm(args, preprocess_config, model_config, train_config)

    Loss = MelLoss().to(device)

    # Init logger，应该暂时不用动
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # Training，训练参数的导入
    step = args.restore_step + 1
    total_step = train_config["step"]["total_step_{}".format(args.model)]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    # 创建进度条相关的内容
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    epoch = 1

    while True:  # 应该是实际上训练的位置
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                # 这里是cm模型的一次训练
                mels = batch[6][:, None, :, :]  # torch.Size([batch_size, 1, seq_len, 80])
                cond_dict = {
                    "speakers": batch[2],  # 2
                    "texts": batch[3],  # 3
                    "src_lens": batch[4],  # 4
                    # "max_src_len": batch[5],  # 5 <class 'numpy.int64'> 108
                    "mel_lens": batch[7],  # 7
                    # "max_mel_len": batch[8],  # 8 <class 'numpy.int64'> 645
                    'pitch': batch[9]['pitch'],
                    'f0': batch[9]['f0'],
                    'uv': batch[9]['uv'],
                    'cwt_spec': batch[9]['cwt_spec'],
                    'f0_mean': batch[9]['f0_mean'],
                    'f0_std': batch[9]['f0_std'],
                    "e_targets": batch[10],  #10
                    "d_targets": batch[11],
                    "mel2phs": batch[12],
                    "spker_embeds": batch[13],  # 为了和diffgan融合增加的
                }
                # cond_dict["mel_masks"] = get_mask_from_lengths(cond_dict["mel_lens"], mels.size(2))

                cm_loss = cm_train_tool.run_step(batch=mels, cond=cond_dict)

                # 分模型进行log
                if step % log_step == 0:
                    # 在评估的时候才进行一个预测,然后计算梅尔loss
                    with torch.no_grad():
                        mel_predictions = cm_train_tool.synthesize_step(batch=mels, cond=cond_dict)
                        mel_loss = Loss(
                            mels[:, 0], mel_predictions,
                            # src_lens=cond_dict["src_lens"],  # 4
                            # max_src_len=cond_dict["texts"].size(1),  # 5
                            mel_lens=cond_dict["mel_lens"],
                            max_mel_len=mels.size(2),
                        )
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Mel Loss:"+str(mel_loss) + "CM Loss: ,"+str(cm_loss)

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    # log_cm(train_logger, step, losses=losses, lr=cm_train_tool.get_lr())
                    pass

                # 分模型进行结果输出
                if step % synth_step == 0:
                    pass

                if step % val_step == 0:
                    pass

                if step % save_step == 0:
                    # 保持文件安全
                    cm_train_tool.save()  # 直接调用这个函数就可以保存CM部分了
                    torch.cuda.empty_cache()

                if step >= total_step:
                    # 因为是死循环，所以在这里大于全部的训练步数时候停止
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow",
                 "diff_naive", "diff_aux", "diff_shallow",  # 扩散模型
                 "consistency_training", "consistency_distillation",  # 一致性模型
                 ],
        required=True,
        help="training model type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    # print(train_config)
    configs = (preprocess_config, model_config, train_config)
    if args.model in ["shallow", "diff_shallow"]:
        assert args.restore_step >= train_config["step"]["total_step_aux"]
    if args.model in ["aux", "shallow", "diff_aux", "diff_shallow"]:
        train_tag = "shallow"
    elif args.model in ["naive", "diff_naive"]:
        train_tag = "naive"
    elif args.model in ["consistency_training", "consistency_distillation"]:
        train_tag = "cm"
    else:
        raise NotImplementedError
    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    # train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"]+"_{}{}".format(train_tag, path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"]+"_{}{}".format(train_tag, path_tag)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(" ---> Type of Modeling:", args.model)
    if model_config["multi_speaker"]:
        print(" ---> Type of Speaker Embedder:", preprocess_config["preprocessing"]["speaker_embedder"])
    print(" ---> Total Batch Size:", int(train_config["optimizer"]["batch_size"]))
    print(" ---> Use Pitch Embed:", model_config["variance_embedding"]["use_pitch_embed"])
    print(" ---> Use Energy Embed:", model_config["variance_embedding"]["use_energy_embed"])
    print(" ---> Path of ckpt:", train_config["path"]["ckpt_path"])
    print(" ---> Path of log:", train_config["path"]["log_path"])
    print(" ---> Path of result:", train_config["path"]["result_path"])
    print("================================================================================================")

    main(args, configs)