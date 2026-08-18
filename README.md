# Overview
This repository contains the implementation code for the paper **"DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization"** by [Christos Koutlis](https://orcid.org/0000-0003-3682-408X) and [Symeon Papadopoulos](https://orcid.org/0000-0002-5441-7341)), available at [arXiv:2411.10193](https://arxiv.org/abs/2411.10193).
![dimodif](https://github.com/mever-team/dimodif/blob/main/architecture.png)

>Deepfake technology has rapidly advanced and poses significant threats to information integrity and trust in online multimedia. While significant progress has been made in detecting deepfakes, the simultaneous manipulation of audio and visual modalities, sometimes at small parts or in subtle ways, presents highly challenging detection scenarios. To address these challenges, we present DiMoDif, an audio-visual deepfake detection framework that leverages the inter-modality differences in machine perception of speech, based on the assumption that in real samples -- in contrast to deepfakes -- visual and audio signals coincide in terms of information. DiMoDif leverages features from deep networks that specialize in visual and audio speech recognition to spot frame-level cross-modal incongruities, and in that way to temporally localize the deepfake forgery. To this end, we devise a hierarchical cross-modal fusion network, integrating adaptive temporal alignment modules and a learned discrepancy mapping layer to explicitly model the subtle differences between visual and audio representations. Then, the detection model is optimized through a composite loss function accounting for frame-level detections and fake intervals localization. DiMoDif outperforms the state-of-the-art on the Deepfake Detection task by 30.5 AUC on the highly challenging AV-Deepfake1M, while it performs exceptionally on FakeAVCeleb and LAV-DF. On the Temporal Forgery Localization task, it outperforms the state-of-the-art by 47.88 AP@0.75 on AV-Deepfake1M, and performs on-par on LAV-DF.


# Setup
* Clone the repository:
   ```
   git clone https://github.com/mever-team/dimodif
   ```

* Create the directories `utils` and `data`:
   ```
   cd dimodif
   mkdir utils
   mkdir data
   ```

* Create the environment:
   ```
   conda create -n dimodif python=3.10
   conda activate dimodif
   pip install -r requirements.txt
   conda install conda-forge::sox
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

* Install `ffmpeg` (for robustness experiments only):
   ```
   sudo apt update
   sudo apt install ffmpeg
   ```

* Check that it is placed in `/usr/bin/ffmpeg` with `which ffmpeg`.

* Download the datasets [FakeAVCeleb](https://sites.google.com/view/fakeavcelebdash-lab/), [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html), [LAV-DF](https://github.com/ControlNet/LAV-DF), [AV-Deepfake1M](https://deepfakes1m.github.io/), [DFDC](https://ai.meta.com/datasets/dfdc/), [KoDF](https://deepbrainai-research.github.io/kodf/).

* Extract the corresponding features using [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) ASR and VSR models:
   * Clone the repository providing the speech recognition backbones:
      ```
      git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
      ```
   * Download the visual-only (`LRS3_V_WER19.1`) and audio-only (`LRS3_A_WER1.0`) models from the links provided in https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models and unzip them in `Visual_Speech_Recognition_for_Multiple_Languages/data`

   * Copy Python files for language identification and feature extraction:
      ```
      cp external/voxceleb2.py Visual_Speech_Recognition_for_Multiple_Languages
      cp external/features.py Visual_Speech_Recognition_for_Multiple_Languages
      ```

   * Prepare data and run
      ```
      cd Visual_Speech_Recognition_for_Multiple_Languages
      mkdir datasets
      ln -s path/to/FakeAVCeleb_v1.2 datasets
      ln -s path/to/VoxCeleb2 datasets
      ln -s path/to/LAV-DF datasets
      ln -s path/to/AV-Deepfake1M datasets
      ln -s path/to/DFDC datasets
      ln -s path/to/KoDF datasets
      python voxceleb2.py  # identifies English speaking females and males
      python features.py -d <dataset_name>  # extracts features
      ```
      `<dataset_name>` accepts the following values: `fakeavceleb`, `voxceleb`, `dfdc`, `lavdf`, `avdeepfake1m`, `kodf`.

* Copy the extracted features of all datasets or create appropriate symlinks in `dimodif/data` directory:
   ```
   ln -s datasets/FakeAVCeleb_emb ../data
   ln -s datasets/VoxCeleb2_emb ../data
   ln -s datasets/LAV-DF_emb ../data
   ln -s datasets/AV-Deepfake1M_emb ../data
   ln -s datasets/DFDC_emb ../data
   ln -s datasets/KoDF_emb ../data
   ```

* Place the corresponding metadata files in these directories:
   ```
   ln -s datasets/FakeAVCeleb_v1.2/meta_data.csv ../data/FakeAVCeleb_emb

   ln -s datasets/LAV-DF/metadata.min.json ../data/LAV-DF_emb

   ln -s datasets/AV-Deepfake1M/train_metadata.json ../data/AV-Deepfake1M_emb

   ln -s datasets/AV-Deepfake1M/val_metadata.json ../data/AV-Deepfake1M_emb

   for i in {0..49}
   do
      cp datasets/DFDC/train/dfdc_train_part_$i/metadata.json ../data/DFDC_emb/train/dfdc_train_part_$i
   done
   cp datasets/DFDC/validation/labels.csv ../data/DFDC_emb/validation
   cp datasets/DFDC/test/labels.csv ../data/DFDC_emb/test
   ```

* For robustness experiments:
  * Copy the `epsnet` folder to `dimodif`:
      ```
      cd ../
      cp -r Visual_Speech_Recognition_for_Multiple_Languages/espnet ./
      ```
   * Copy required files to `ckpt/avsr`:
      ```
      cp -r Visual_Speech_Recognition_for_Multiple_Languages/data/LRS3_A_WER1.0 ckpt/avsr
      cp -r Visual_Speech_Recognition_for_Multiple_Languages/data/LRS3_V_WER19.1 ckpt/avsr
      cp Visual_Speech_Recognition_for_Multiple_Languages/pipelines/tokens/unigram5000_units.txt ckpt/avsr
      cp Visual_Speech_Recognition_for_Multiple_Languages/pipelines/detectors/retinaface/20words_mean_face.npy ckpt/avsr
      ```
   * Create symlinks to the original video data of `FakeAVCeleb` and `VoxCeleb2`:
      ```
      ln -s path/to/FakeAVCeleb_v1.2 data
      ln -s path/to/VoxCeleb2 data
      ```
   * Create the folder `utils/robustness` with `mkdir utils/robustness`

# Experiments
Training experiments are conducted with the following scripts:
* [***Grid***] Deepfake Detection (DFD) on FakeAVCeleb, LAV-DF & AV-Deepfake1M: `scripts/dfd.py`
* [***Best config.***] In-dataset and cross-manipulation (DFD) on FakeAVCeleb: `scripts/favc.py` (creates in-dataset and cross-manipulation checkpoints)
* [***Grid***] Temporal Forgery Localization (TFL) on LAV-DF & AV-Deepfake1M: `scripts/tfl.py`
* [***Best config.***] Ablations: `scripts/ablation.py`

Evaluations are conducted with the following scripts:
* Predictions on AV-Deepfake1M test set to upload to Codabench for evaluation: `scripts/predictions_avdeepfake1m.py`
* In-dataset evaluation, cross-dataset generalization, and cross-manipulation evaluation (FakeAVCeleb): `scripts/eval.py`
* Robustness evaluation experiments are conducted with `scripts/robustness.py`

Checkpoints are generated after retraining the best configurations with `scripts/best.py`.

Paper results obtained with `scripts/results.py`.

# Checkpoints
Deepfake Detection checkpoints:
* [***FakeAVCeleb***] `ckpt/dfd/dfd_fakeavceleb_reduceonplateau_64_4_1_3_True_<forgery-type>.pth`

   `<forgery-type>` determines the checkpoint to use. Replace with `none` for the in-dataset experiments checkpoint. Replace with `rvfa`, `fvra-wl`, `fvfa-fs`, `fvfa-gan`, `fvfa-wl` for the cross-manipulation experiments checkpoints.
* [***LAV-DF***] `ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`
* [***AV-Deepfake1M***] `ckpt/dfd/dfd_avdeepfake1m_reduceonplateau_64_4_1_5_True_whole.pth` (the checkpoint without '_whole' suffix derives from training on partial training dataset of 200K samples - **not for use**)

Temporal Forgery Localization checkpoints:
* [***LAV-DF***] `ckpt/tfl/tfl_lavdf_reduceonplateau_256_8_2_5_True.pth`
* [***AV-Deepfake1M***] `ckpt/tfl/tfl_avdeepfake1m_reduceonplateau_256_8_2_5_True_whole.pth` (the checkpoint without '_whole' suffix derives from training on partial training dataset of 200K samples - **not for use**)

# Citation
```
@article{koutlis2024dimodif,
title={DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization},
author={Koutlis, Christos and Papadopoulos, Symeon},
journal={arXiv preprint arXiv:2411.10193},
year={2024}
}
```

# Contact
Christos Koutlis ([ckoutlis@iti.gr](ckoutlis@iti.gr))
