import argparse

from utils.tools import get_configs_of
from preprocessor import ljspeech, vctk,libritts


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "VCTK" in config["dataset"]:
        vctk.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    config, *_ = get_configs_of(args.dataset)
    main(config)
