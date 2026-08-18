import os
import pandas as pd
import pickle
import torch
import difflib
import json
import numpy as np
import argparse
import traceback
from PIL import Image
from facenet_pytorch import MTCNN
from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector

FAST_FEATURES = os.environ.get("DIMODIF_FAST_FEATURES", "1") != "0"


class InferencePipeline(torch.nn.Module):
    def __init__(
        self,
        modality,
        model_path,
        model_conf,
        detector="mediapipe",
        face_track=False,
        device="cuda:0",
    ):
        super(InferencePipeline, self).__init__()
        self.device = device
        # modality configuration
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(
            modality,
            model_path,
            model_conf,
            rnnlm=None,
            rnnlm_conf=None,
            penalty=0.0,
            ctc_weight=0.1,
            lm_weight=0.0,
            beam_size=40,
            device=device,
        )
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None

    def process_landmarks(self, data_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            landmarks = self.landmarks_detector(data_filename)
            return landmarks

    def forward(self, data_filename, landmarks=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        if landmarks is None:
            landmarks = self.process_landmarks(data_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def extract_features(self, data_filename, landmarks=None, extract_resnet_feats=False):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        data = None
        try:
            if landmarks is None:
                landmarks = self.process_landmarks(data_filename)
            data = self.dataloader.load_data(data_filename, landmarks)
            transcript = None if FAST_FEATURES else self.model.infer(data)
        except Exception as exc:
            print(f"[DiMoDif] Failed to load/infer {self.modality} data from {data_filename}: {exc}", flush=True)
            traceback.print_exc()
            transcript = None

        if data is None:
            return None, transcript, landmarks

        try:
            with torch.no_grad():
                if isinstance(data, tuple):
                    enc_feats = self.model.model.encode(
                        data[0].to(self.device),
                        data[1].to(self.device),
                        extract_resnet_feats,
                    )
                else:
                    enc_feats = self.model.model.encode(data.to(self.device), extract_resnet_feats)
        except Exception as exc:
            print(f"[DiMoDif] Failed to encode {self.modality} data from {data_filename}: {exc}", flush=True)
            traceback.print_exc()
            enc_feats = None

        return enc_feats, transcript, landmarks


def get_mtcnn_landmarks(video_path, pipeline, detector):
    try:
        video = pipeline.dataloader.load_video(video_path)
    except:
        print("Error during video loading.")
        return None
    landmarks = []
    for frame in video:
        _, _, landmarks_ = detector.detect(Image.fromarray(frame), landmarks=True)
        if landmarks_ is not None:
            landmarks_ = landmarks_.astype(int)[0]
            landmarks_[3][0] = int((landmarks_[3][0] + landmarks_[4][0]) / 2)
            landmarks_[3][1] = int((landmarks_[3][1] + landmarks_[4][1]) / 2)
            landmarks_ = landmarks_[:4]
            landmarks.append(landmarks_)
        else:
            landmarks.append(None)
    return landmarks


def inference(pipeline, video_path, detector_name="mediapipe", detector=None):
    if pipeline.modality == "video":
        if detector_name == "mediapipe":
            landmarks = None
        elif detector_name == "mtcnn":
            landmarks = get_mtcnn_landmarks(video_path, pipeline, detector)
        else:
            raise Exception(f"{detector_name} detector not supported.")
    else:
        landmarks = None
    features, transcript, landmarks = pipeline.extract_features(video_path, landmarks=landmarks)
    metadata = getattr(pipeline.dataloader, "metadata", None) or {}
    if not metadata:
        metadata = {"data_filename": video_path, "modality": pipeline.modality}
    return features, transcript, landmarks, metadata


def save_data(data, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    np.savez(
        os.path.join(outdir, "features.npz"),
        video_features=data["video_features"],
        audio_features=data["audio_features"],
    )
    with open(os.path.join(outdir, "metadata.json"), "w") as handle:
        json.dump(data["metadata"], handle)


def has_valid_features(outdir):
    feature_path = os.path.join(outdir, "features.npz")
    metadata_path = os.path.join(outdir, "metadata.json")
    if not os.path.exists(feature_path) or not os.path.exists(metadata_path):
        return False
    try:
        data = np.load(feature_path, allow_pickle=True)
        return (
            "video_features" in data.files
            and "audio_features" in data.files
            and data["video_features"].shape != ()
            and data["audio_features"].shape != ()
        )
    except Exception:
        return False


def extract_data(pipeline_audio, pipeline_video, video_path, detector_name="mediapipe", detector=None):
    audio_features, audio_transcript, _, audio_metadata = inference(
        pipeline=pipeline_audio,
        video_path=video_path,
        detector_name=detector_name,
        detector=detector,
    )
    video_features, video_transcript, video_landmarks, video_metadata = inference(
        pipeline=pipeline_video,
        video_path=video_path,
        detector_name=detector_name,
        detector=detector,
    )
    if (video_landmarks is not None) and (video_features is not None) and (audio_features is not None):
        metadata = {**audio_metadata, **video_metadata}
        metadata["audio_transcript"] = audio_transcript
        metadata["video_transcript"] = video_transcript
        metadata["num_frames_no_landmarks"] = sum([x is None for x in video_landmarks])
        metadata["transcript_video_num_char"] = len(video_transcript) if video_transcript is not None else None
        metadata["transcript_audio_num_char"] = len(audio_transcript) if audio_transcript is not None else None
        metadata["num_diffs"] = (
            len([x for x in difflib.ndiff(video_transcript, audio_transcript) if x[0] != " "])
            if (video_transcript is not None) and (audio_transcript is not None)
            else None
        )
        metadata["diff_score"] = (
            (metadata["num_diffs"] / (metadata["transcript_video_num_char"] + metadata["transcript_audio_num_char"]))
            if metadata["num_diffs"] is not None
            else None
        )
        data = {
            "audio_features": audio_features.detach().cpu().numpy(),
            "video_features": video_features.detach().cpu().numpy(),
            "metadata": metadata,
        }
    else:
        data = {
            "audio_features": None,
            "video_features": None,
            "metadata": None,
        }

    return data


def load_data(filename):
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    return data


def inpath2outdir(dataset, infile, detector_name):
    if dataset == "fakeavceleb":
        return os.path.join(
            os.path.splitext(infile.replace("datasets/FakeAVCeleb_v1.2", "datasets/FakeAVCeleb_emb"))[0],
            detector_name,
        )
    elif dataset == "voxceleb":
        return os.path.join(
            os.path.splitext(infile.replace("datasets/VoxCeleb2", "datasets/VoxCeleb2_emb"))[0],
            detector_name,
        )
    elif dataset == "dfdc":
        return os.path.join(
            os.path.splitext(infile.replace("datasets/DFDC", "datasets/DFDC_emb"))[0],
            detector_name,
        )
    elif dataset == "lavdf":
        return os.path.join(
            os.path.splitext(infile.replace("datasets/LAV-DF", "datasets/LAV-DF_emb"))[0],
            detector_name,
        )
    elif dataset == "avdeepfake1m":
        return os.path.join(
            os.path.splitext(
                infile.replace(
                    "datasets/AV-Deepfake1M",
                    "datasets/AV-Deepfake1M_emb",
                )
            )[0],
            detector_name,
        )
    elif dataset == "hifi_avdf":
        return os.path.join(
            os.path.splitext(
                infile.replace(
                    "datasets/HiFi-AVDF",
                    "datasets/HiFi-AVDF_emb",
                )
            )[0],
            detector_name,
        )
    elif dataset == "kodf":
        return os.path.join(
            os.path.splitext(
                infile.replace(
                    "datasets/KoDF",
                    "datasets/KoDF_emb",
                )
            )[0],
            detector_name,
        )
    else:
        raise Exception(f"{dataset} dataset is not supported.")


def get_video_paths(dataset, split, bounds):
    if dataset == "fakeavceleb":
        datdir = "datasets/FakeAVCeleb_v1.2"
        metadata = pd.read_csv(os.path.join(datdir, "meta_data.csv"))
        paths = [
            os.path.join(x[0], x[1]).replace("FakeAVCeleb", datdir)
            for x in metadata[["Unnamed: 9", "path"]].values.tolist()
        ]
    elif dataset == "voxceleb":
        with open("voxceleb2_eng.json", "r") as hundle:
            paths = json.load(hundle)
    elif dataset == "dfdc":
        train_dir = "datasets/DFDC/train"
        val_dir = "datasets/DFDC/validation"
        test_dir = "datasets/DFDC/test"
        metadata = {}
        for i in range(50):
            with open(f"{train_dir}/dfdc_train_part_{i}/metadata.json") as f:
                metadata.update(json.load(f))
        paths = []
        if split in ["train", "all"]:
            train_paths_r = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(train_dir, topdown=False)
                for name in files
                if (".mp4" in name) and (metadata[name]["label"] == "REAL")
            ]
            train_paths_f = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(train_dir, topdown=False)
                for name in files
                if (".mp4" in name) and (metadata[name]["label"] == "FAKE")
            ]
            paths.extend(train_paths_r + train_paths_f)
        if split in ["val", "all"]:
            val_paths = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if ".mp4" in x]
            paths.extend(val_paths)
        if split in ["test", "all"]:
            test_paths = [os.path.join(test_dir, x) for x in os.listdir(test_dir) if ".mp4" in x]
            paths.extend(test_paths)
    elif dataset == "lavdf":
        directory = "datasets/LAV-DF"
        train_dir = os.path.join(directory, "train")
        dev_dir = os.path.join(directory, "dev")
        test_dir = os.path.join(directory, "test")
        paths = []
        if split in ["train", "all"]:
            train_paths = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(train_dir, topdown=False)
                for name in files
                if ".mp4" in name
            ]
            paths.extend(train_paths)
        if split in ["val", "all"]:
            dev_paths = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(dev_dir, topdown=False)
                for name in files
                if ".mp4" in name
            ]
            paths.extend(dev_paths)
        if split in ["test", "all"]:
            test_paths = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(test_dir, topdown=False)
                for name in files
                if ".mp4" in name
            ]
            paths.extend(test_paths)
    elif dataset == "avdeepfake1m":
        directory = "datasets/AV-Deepfake1M"
        train_dir = os.path.join(directory, "train")
        val_dir = os.path.join(directory, "val")
        test_dir = os.path.join(directory, "test")

        paths = []
        if split in ["train", "all"]:
            train_paths = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(train_dir, topdown=False)
                for name in files
                if ".mp4" in name
            ]
            paths.extend(train_paths)
        if split in ["val", "all"]:
            val_paths = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(val_dir, topdown=False)
                for name in files
                if ".mp4" in name
            ]
            paths.extend(val_paths)
        if split in ["test", "all"]:
            test_paths = [
                os.path.join(root, name)
                for root, dirs, files in os.walk(test_dir, topdown=False)
                for name in files
                if ".mp4" in name
            ]
            paths.extend(test_paths)
    elif dataset == "kodf":
        directory = "datasets/KoDF/"
        realdir = os.path.join(directory, "original_videos")
        fakedir_ad = os.path.join(directory, "synthesized_videos", "audio-driven")
        fakedir_dffs = os.path.join(directory, "synthesized_videos", "dffs")
        fakedir_dfl = os.path.join(directory, "synthesized_videos", "dfl")
        fakedir_fo = os.path.join(directory, "synthesized_videos", "fo")
        fakedir_fsgan = os.path.join(directory, "synthesized_videos", "fsgan")
        np.random.seed(0)
        real_paths = np.random.choice(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(realdir, topdown=False)
                for name in files
                if ".mp4" in name
            ],
            1000,
        ).tolist()
        fake_ad_paths = np.random.choice(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(fakedir_ad, topdown=False)
                for name in files
                if ".mp4" in name
            ],
            1000,
        ).tolist()
        fake_dffs_paths = np.random.choice(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(fakedir_dffs, topdown=False)
                for name in files
                if ".mp4" in name
            ],
            1000,
        ).tolist()
        fake_dfl_paths = np.random.choice(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(fakedir_dfl, topdown=False)
                for name in files
                if ".mp4" in name
            ],
            1000,
        ).tolist()
        fake_fo_paths = np.random.choice(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(fakedir_fo, topdown=False)
                for name in files
                if ".mp4" in name
            ],
            1000,
        ).tolist()
        fake_fsgan_paths = np.random.choice(
            [
                os.path.join(root, name)
                for root, dirs, files in os.walk(fakedir_fsgan, topdown=False)
                for name in files
                if ".mp4" in name
            ],
            1000,
        ).tolist()
        paths = real_paths + fake_ad_paths + fake_dffs_paths + fake_dfl_paths + fake_fo_paths + fake_fsgan_paths
    elif dataset == "hifi_avdf":
        directory = "datasets/HiFi-AVDF/"
        generators = ["kling2.5", "veo3.1", "wan2.5", "seedance1.0"]
        paths = [
            os.path.join(directory, generator, name)
            for generator in generators
            for name in sorted(os.listdir(os.path.join(directory, generator)))
            if name.endswith(".mp4")
        ]
    else:
        raise Exception(f"{dataset} dataset is not supported.")

    if bounds:
        return paths[bounds[0] : bounds[1]]
    else:
        return paths


def get_pipeline_obj(modality, gpu):
    if modality == "audio":
        pipeline = InferencePipeline(
            modality="audio",
            model_path="data/LRS3_A_WER1.0/model.pth",
            model_conf="data/LRS3_A_WER1.0/model.json",
            face_track=False,
            device=f"cuda:{gpu}",
        )
        print(
            f"[DiMoDif] {modality} model device={next(pipeline.model.model.parameters()).device}; "
            f"cuda_allocated={torch.cuda.memory_allocated(int(gpu)) / 1024**3:.2f}GB",
            flush=True,
        )
        return pipeline
    elif modality == "video":
        pipeline = InferencePipeline(
            modality="video",
            model_path="data/LRS3_V_WER19.1/model.pth",
            model_conf="data/LRS3_V_WER19.1/model.json",
            face_track=True,
            device=f"cuda:{gpu}",
        )
        print(
            f"[DiMoDif] {modality} model device={next(pipeline.model.model.parameters()).device}; "
            f"cuda_allocated={torch.cuda.memory_allocated(int(gpu)) / 1024**3:.2f}GB",
            flush=True,
        )
        return pipeline
    else:
        raise Exception(f"{modality} modality is not supported.")


def store_videos_embeddings(
    dataset,
    video_paths,
    pipeline_audio,
    pipeline_video,
    detector_name="mediapipe",
    detector=None,
):
    n = len(video_paths)
    for i, video_path in enumerate(video_paths):
        outdir = inpath2outdir(dataset, video_path, detector_name)
        print(f"{i + 1}/{n} {video_path} --> {outdir}")
        if not has_valid_features(outdir):
            data = extract_data(
                pipeline_audio,
                pipeline_video,
                video_path,
                detector_name=detector_name,
                detector=detector,
            )
            save_data(data, outdir)


def main(dataset, detector_name, split, bounds, gpu):
    video_paths = get_video_paths(dataset, split, bounds)
    pipeline_audio = get_pipeline_obj("audio", gpu)
    pipeline_video = get_pipeline_obj("video", gpu)
    if detector_name == "mediapipe":
        detector = None
    elif detector_name == "mtcnn":
        detector = MTCNN(device=f"cuda:{gpu}")
    else:
        raise Exception(f"{detector_name} detector not supported.")
    store_videos_embeddings(
        dataset,
        video_paths,
        pipeline_audio,
        pipeline_video,
        detector_name=detector_name,
        detector=detector,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVSR feature extraction for AV Deepfake detection.")
    parser.add_argument(
        "-d",
        "--dataset_name",
        help="the name of the dataset to extract features from",
        choices=["fakeavceleb", "voxceleb", "dfdc", "lavdf", "avdeepfake1m", "kodf", "hifi_avdf"],
    )
    parser.add_argument(
        "-l",
        "--detector_name",
        help="the name of the facial landmarks detector",
        choices=["mediapipe", "mtcnn"],
        default="mediapipe",
    )
    parser.add_argument(
        "-s",
        "--split",
        help="determines the dataset split to extract features from",
        choices=["train", "val", "test", "all"],
        default="all",
    )
    parser.add_argument(
        "-b",
        "--bounds",
        help="determines a part of the dataset to extract features from, in the form [x,y]",
        default="[]",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help="determines the GPU device to use",
        default="0",
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset_name,
        detector_name=args.detector_name,
        split=args.split,
        bounds=eval(args.bounds),
        gpu=args.gpu,
    )
