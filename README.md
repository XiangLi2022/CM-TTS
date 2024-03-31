## CM-TTS: Enhancing Real Time Text-to-Speech Synthesis Efficiency through Weighted Samplers and Consistency Models

Generated samples are accessible through the following link: [Research showcase](https://cmtts.vercel.app/)


# Quickstart

---

DATASET refers to the names of datasets such as LibriTTS and VCTK in the following documents.

---

## Dependencies
You can install the Python dependencies with:

`
pip3 install -r requirements.txt
`

---

## Synthesize
You have to download the pretrained models(To prevent anonymity from leaking, we share the link later) and put them in`output/pretrained_model/DATASET/CMDenoiserTTS/`.

<!-- [pretrained models](https://drive.google.com/drive/folders/1DKjEyDeHvOvm9qvedaO8LFET573gpMt6?usp=drive_link) -->

### Synthesize by Single Text


Synthesize on VCTK.
`
bash single_synthesize_vctk.sh
`

Synthesize on LJSpeech.
`
bash single_synthesize_lj.sh
`

Synthesize on LibriTTS.
`
bash single_synthesize_lib.sh
`


### Synthesize by Single Batch
Synthesize on VCTK.
`
bash synthesize_vctk.sh
`

Synthesize on LJSpeech.
`
bash synthesize_lj.sh
`

Predict on LibriTTS.
`
bash synthesize_lib.sh
`
### Synthesize Zeroshot Sample
You can achieve zero-shot synthesis across datasets using the following approach:

Train on the LibriTTS dataset and predict on VCTK.
`
bash synthesize_lib2vctk.sh
`

Train on the LibriTTS dataset and predict on LJSpeech.
`
bash synthesize_lib2lj.sh
`

# Training

---

## Preprocessing Data

For a multi-speaker TTS with external speaker embedder, download [ResCNN Softmax+Triplet pretrained model](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP/edit) of philipperemy's DeepSpeaker for the [speaker embedding](https://github.com/philipperemy/deep-speaker) and locate it in ./deepspeaker/pretrained_models/.

For the forced alignment, [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/) is used to obtain the alignments between the utterances and the phoneme sequences. Pre-extracted alignments for the datasets are provided [here](https://drive.google.com/drive/folders/1fizpyOiQ1lG2UDaMlXnT3Ll4_j6Xwg7K). You have to unzip the files in preprocessed_data/DATASET/TextGrid/. Alternately, you can [run the aligner by yourself](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/index.html).

Before starting the processing, please check if `.\config\DATASET\preprocess.yaml` has been configured according to your preferences.

After completing the above preparations, you can accomplish data processing by running the corresponding script. The processing script is as follows:

LJSpeech:
`
bash deal_data_Lj.sh
`

VCTK:
`
bash deal_data_VCTK.sh
`


LibriTTS:
`
bash deal_data_Lib.sh
`

---

## Start Train
Before starting the training, please ensure that all the configurations under `.\config\DATASET` have been set according to your preferences, and the data has been processed as described above. 

You can perform training on different datasets.

LJSpeech:
`
python3 train_cm.py --model consistency_training  --dataset LJSpeech
`

VCTK:
`
python3 train_cm.py --model consistency_training  --dataset VCTK
`

LibriTTS:
`
python3 train_cm.py --model consistency_training  --dataset LibriTTS
`


# Supplementary Experiments
**Experiment Description for MOS**:In this supplementary experiment, we 
added the MOS metric test results for all models on the VCTK and 
LJSpeech datasets to Tables 1 and 2. For Table 3 and 4, we added the MOS test results for CM-TTS and 
DiffGAN-TTS models in a zero-shot scenario. For Table 5, we added the MOS metric 
test results for CM-TTS under different sampler settings. For Table 6, we 
added the MOS metric test results for CM-TTS under different loss settings. 
The specific results are as 
follows.

####  Tables 1:MOS on VCTK dataset
|      Models       |         MOS         |
|:-----------------:|:-------------------:|
| Reference (voc.)  |   4.5826(±0.1147)   |
| FastSpeech2(300K) |   3.6821(±0.1762)   |
|       VITS        |   3.6717(±0.0123)   |
|  DiffSpeech       |   2.9157(±0.0594)   |
| DiffGAN-TTS(T=1)  |   3.4476(±0.1038)   |
| DiffGAN-TTS(T=2)  |   3.6173(±0.1433)   |
| DiffGAN-TTS(T=4)  |   3.6143(±0.1186)   |
|    CM-TTS(T=1)    | **3.9618(±0.0186)** |
|    CM-TTS(T=2)    |   3.8947(±0.0262)   |
|    CM-TTS(T=4)    |   3.8623(±0.0311)   |

#### Tables 2:MOS on LJSpeech dataset
| Models                |         MOS         |
|:----------------------|:-------------------:|
| Reference (voc.)      |   4.8667(±0.0315)   |
| FastSpeech2(300K)     |   3.5742(±0.2309)   |
| DiffSpeech            |   3.1668(±0.1378)   |
| CoMoSpeech            |   3.5583(±0.2421)   |
| VITS                  |   3.6234(±0.0252)   |
| DiffGAN-TTS(T=1)      |   3.7142(±0.1390)   |
| DiffGAN-TTS(T=2)      |   3.6813(±0.0561)   |
| DiffGAN-TTS(T=4)      |   3.7258(±0.0087)   |
| CM-TTS(T=1)           | **3.8353(±0.0179)** |
| CM-TTS(T=2)           |   3.7917(±0.1356)   |
| CM-TTS(T=4)           |   3.7602(±0.1327)   |

#### Tables 3:MOS on VCTK under Zero-Shot Setting
|        Models        |         MOS         |
|:-------------------:|:-------------------:|
| Reference (voc.)    |   4.7467(±0.0194)   |
|  DiffGAN-TTS(T=1)   |   3.4607(±0.1880)   |
|  DiffGAN-TTS(T=2)   |   3.5067(±0.1573)   |
|  DiffGAN-TTS(T=4)   |   3.5893(±0.0298)   |
|    CM-TTS(T=1)      |   3.8715(±0.0896)   |
|    CM-TTS(T=2)      |   3.8387(±0.1521)   |
|    CM-TTS(T=4)      | **3.9221(±0.1016)** |

#### Tables 4:MOS on LJSpeech under Zero-Shot Setting
| Models            |       MOS        |
|:------------------|:----------------:|
| Reference (voc.)  | 4.8832(±0.0174)  |
| DiffGAN-TTS(T=1)  | 3.6047(±0.1015)  |
| DiffGAN-TTS(T=2)  | 3.6212(±0.0771)  |
| DiffGAN-TTS(T=4)  | 3.7361(±0.1802)  |
| CM-TTS(T=1)       | 3.7205(±0.1097)  |
| CM-TTS(T=2)       | 3.6817(±0.1328)  |
| CM-TTS(T=4)       | 3.7113(±0.1022)  |


#### Tables 5:MOS on VCTK with Different Samplers
| Types              |         MOS         |
|:-------------------|:-------------------:|
| Reference (voc.)   |   4.7172(±0.1236)   |
| Uniform            |   3.8133(±0.0727)   |
| Linear(↗)          |   3.3278(±0.0803)   |
| Linear(↘)          |   3.5676(±0.1488)   |
| LSM                | **3.9107(±0.1254)** |

#### Tables 6:MOS on VCTK with Different Loss
| Types              |         MOS         |
|:-------------------|:-------------------:|
| Reference (voc.)   |   4.6304(±0.1418)   |
| L1                 | **3.9052(±0.0415)** |
| L1 (w/o padding)   |   3.8117(±0.1005)   |
| L2                 |   3.8726(±0.1971)   |
| L2 (w/o padding)   |   3.8604(±0.1436)   |

Earlier implementations of the FastSpeech 2 model relied on directly importing checkpoints, possibly resulting in loading errors. We have now retrained the model and reassessed the relevant metrics. After carefully re-checking, the updated metrics are now available in the table below.

#### Tables 7:Updated Metrics for FastSpeech2 on VCTK and LJSpeech
| Dataset       | FFE(↓) | Cos-speaker(↑) | mfccFID(↓) | melFID(↓) | mfccRecall(↑) | MCD(↓) | SSIM(↑) | mfccCOS(↑) | F0-RMSE(↓) | wer          |
|:--------------|:------:|:-------------:|:----------:|:---------:|:-------------:|:------:|:-------:|:----------:|:----------:|-------------:|
| FastSpeech2(VCTK)       | 0.3503 |     0.8236    |   43.4236  |   8.8175  |    0.3554     | 5.8897 |  0.4537 |   0.7565   |  119.2076  | 0.0677  |
| FastSpeech2(LJSpeech)         | 0.4877 |     0.8825    |   36.3090   |   5.2796  |    0.2121    | 6.1157 |  0.6468 |   0.7985   |  135.2583  | 0.0944  |

To verify the individual contributions of CT and LSM to the model's performance, we conducted ablation experiments by separately removing CT and LSM. The experimental results are presented below. 

#### Tables 8:Ablation Study on VCTK (T=1)
| Models |   FFE(↓)   | Cos-speaker(↑) | mfccFID(↓) | melFID(↓) | mfccRecall(↑) |  MCD(↓)  |  SSIM(↑)   | mfccCOS(↑) | F0-RMSE(↓) |    WER(↓)     |
|:-------|:----------:|:--------------:|:----------:|:---------:|:-------------:|:--------:|:----------:|:----------:|:----------:|:----------:|
| CM-T1  | **0.3387** |   **0.8396**   | **39.17**  | **7.58**  |    0.3946     | **5.91** | **0.4772** | **0.7599** |   119.29   | **0.0688** |
| -CT    |   0.3364   |    0.835074    |  43.1316   | 10.74238  |    0.40103    |  5.9821  |   0.4626   |   0.7545   | 122.69101  |   0.0832   |
| -LSM   |   0.3351   |     0.8333     |   56.31    |   10.08   |  **0.4015**   |   5.98   |   0.4396   |   0.7456   | **118.87** |   0.0872   |

To further explore the generalization of LSM, we apply it to DiffGAN. The experimental results, as shown in the following table, strongly demonstrate that LSM can bring significant improvements across most metrics.

#### Tables 9:DiffGAN with and without LSM
|       Model       | FFE(↓) | Cos-speaker(↑) | mfccFID(↓) | melFID(↓) | mfccRecall(↑) | MCD(↓) | SSIM(↑) | mfccCOS(↑) | F0-RMSE(↓) |   wer   |
|:-----------------:|:------:|:-------------:|:----------:|:---------:|:-------------:|:------:|:-------:|:----------:|:----------:|:------:|
|   ground_truth    | 0.1427 |     0.9424    |   31.9789  |   3.4802  |     0.5644    | 4.567  |  0.8132 |   0.8457   |   89.2136  | 0.0412 |
|    diff-ganT2     | 0.3411 |     0.8333    |   38.6428  |   7.7855  |     0.3974    | 5.9437 |  0.461  |   0.7581   |   117.1919 | 0.0827 |
| +LSM    | 0.3397 |     0.8397    |   42.9622  |   7.9161  |     0.399     | 5.8576 |  0.458  |   0.7582   |   115.3769 | 0.072  |
|    diff-ganT4     | 0.3465 |     0.8358    |   37.1099  |   6.5823  |     0.3662    | 5.9425 |  0.4614 |   0.7571   |   120.0975 | 0.0751 |
| +LSM    | 0.34054|     0.8403    |   43.8128  |   7.8876  |     0.387     | 5.8742 |  0.4641 |   0.759    |   115.8887 | 0.0704 |
