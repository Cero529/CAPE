from torch.utils.data import Dataset
import torch
import os
import numpy as np
import json
import pandas as pd
import random


class FakeAVCeleb(Dataset):

    def __init__(self, split, max_length, without="none", backbone="ma22", showsize=True):
        fake_category = {
            "rvfa": {"method": "rtvc", "type": "RealVideo-FakeAudio"},
            "fvra-wl": {"method": "wav2lip", "type": "FakeVideo-RealAudio"},
            "fvfa-fs": {"method": "faceswap-wav2lip", "type": "FakeVideo-FakeAudio"},
            "fvfa-gan": {"method": "fsgan-wav2lip", "type": "FakeVideo-FakeAudio"},
            "fvfa-wl": {"method": "wav2lip", "type": "FakeVideo-FakeAudio"},
        }
        self.name = "fakeavceleb"
        self.backbone = backbone
        json_filepath_favc = f"utils/fakeavceleb7030.json"
        json_filepath_vox = f"utils/vox.json"
        if os.path.exists(json_filepath_favc) and os.path.exists(json_filepath_vox):
            with open(json_filepath_favc, "r") as hundle:
                self.favc = json.load(hundle)
            with open(json_filepath_vox, "r") as hundle:
                self.vox = json.load(hundle)
        else:
            with open("data/FakeAVCeleb_emb/meta_data.csv", "r") as f:
                metadata = pd.read_csv(f)
                method = {}
                for index, row in metadata.iterrows():
                    row["filename"] = row["filename"].replace(".mp4", "/mediapipe/features.npz")
                    row["path"] = row["path"].replace("FakeAVCeleb", "data/FakeAVCeleb_emb")
                    method[row["path"] + "/" + row["filename"]] = (row["method"], row["type"])
            fvfa_dir = "data/FakeAVCeleb_emb/FakeVideo-FakeAudio"
            fvra_dir = "data/FakeAVCeleb_emb/FakeVideo-RealAudio"
            rvfa_dir = "data/FakeAVCeleb_emb/RealVideo-FakeAudio"
            rvra_dir = "data/FakeAVCeleb_emb/RealVideo-RealAudio"
            voxceleb2_dir = "data/VoxCeleb2_emb"
            fvfa = [
                (os.path.join(root, name), 1, 1) + method[os.path.join(root, name)]
                for root, dirs, files in os.walk(fvfa_dir, topdown=False)
                for name in files
                if name == "features.npz"
            ]
            fvra = [
                (os.path.join(root, name), 1, 0) + method[os.path.join(root, name)]
                for root, dirs, files in os.walk(fvra_dir, topdown=False)
                for name in files
                if name == "features.npz"
            ]
            rvfa = [
                (os.path.join(root, name), 0, 1) + method[os.path.join(root, name)]
                for root, dirs, files in os.walk(rvfa_dir, topdown=False)
                for name in files
                if name == "features.npz"
            ]
            rvra = [
                (os.path.join(root, name), 0, 0) + method[os.path.join(root, name)]
                for root, dirs, files in os.walk(rvra_dir, topdown=False)
                for name in files
                if name == "features.npz"
            ]
            self.favc = fvfa + fvra + rvfa + rvra
            self.vox = [
                (os.path.join(root, name), 0, 0, "real", "RealVideo-RealAudio")
                for root, dirs, files in os.walk(voxceleb2_dir, topdown=False)
                for name in files
                if name == "features.npz"
            ]
            with open(json_filepath_favc, "w") as hundle:
                json.dump(self.favc, hundle, indent=2)
            with open(json_filepath_vox, "w") as hundle:
                json.dump(self.vox, hundle, indent=2)

        random.seed(0)
        random.shuffle(self.favc)
        ntrain = int(len(self.favc) * 0.65)
        nval = int(len(self.favc) * 0.7)
        if split == "train":
            self.videos = self.favc[:ntrain] + self.vox[:13371]
            if without != "none":
                fake_to_delete = [
                    x
                    for x in self.videos
                    if (x[3], x[4]) == (fake_category[without]["method"], fake_category[without]["type"])
                ]
                real_to_delete = [x for x in self.videos if x[3] == "real"][-len(fake_to_delete) :]
                self.videos = [x for x in self.videos if x not in fake_to_delete + real_to_delete]
        elif split == "val":
            self.videos = self.favc[ntrain:nval]
            if without != "none":
                self.videos = [
                    x
                    for x in self.videos
                    if (x[3], x[4]) == (fake_category[without]["method"], fake_category[without]["type"])
                ] + [x for x in self.videos if x[3] == "real"]
        else:
            self.videos = self.favc[nval:]
            if without != "none":
                self.videos = [
                    x
                    for x in self.videos
                    if (x[3], x[4]) == (fake_category[without]["method"], fake_category[without]["type"])
                ] + [x for x in self.videos if x[3] == "real"]
        self.max_length = max_length

        if showsize:
            print(f"\n{split} - without {without}")
            methods = list(set([x[3] for x in self.videos]))
            types = list(set([x[4] for x in self.videos]))
            methods.sort()
            types.sort()
            print("\nMethods:")
            for m in methods:
                print(f"{m}: {len([x[3] for x in self.videos if x[3] == m])}")
            print("\nTypes:")
            for t in types:
                print(f"{t}: {len([x[4] for x in self.videos if x[4] == t])}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            data_path, video_target, audio_target, _, _ = self.videos[idx]
            if self.backbone == "avhubert":
                data_path = (
                    data_path.replace("data/FakeAVCeleb_emb", "data/FakeAVCeleb_emb_avhubert")
                    .replace("data/VoxCeleb2_emb", "data/VoxCeleb2_emb_avhubert")
                    .replace("mediapipe/", "")
                )
            data = np.load(data_path, allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat(
                [video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])]
            )
            audio_features = torch.concat(
                [audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])]
            )

            return [video_features, audio_features, video_target, audio_target]
        except:
            return None


class LAVDF(Dataset):

    def __init__(self, task, split, max_length, backbone="ma22", showsize=True):
        self.name = "lavdf"
        self.backbone = backbone
        if split == "val":
            split = "dev"
        self.task = task
        if os.path.exists(f"utils/lavdf_{split}.json"):
            with open(f"utils/lavdf_{split}.json", "r") as hundle:
                self.videos = json.load(hundle)
        else:
            directory = f"data/LAV-DF_emb/{split}"
            with open("data/LAV-DF_emb/metadata.min.json", "r") as f:
                metadata = {
                    x["file"]: (int(x["modify_video"]), int(x["modify_audio"]), x["fake_periods"]) for x in json.load(f)
                }
            self.videos = []
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    if name == "features.npz":
                        fn = "/".join(os.path.dirname(root).split("/")[-2:]) + ".mp4"
                        video_target, audio_target, fake_periods = metadata[fn]
                        self.videos.append((os.path.join(root, name), video_target, audio_target, fake_periods))
            with open(f"utils/lavdf_{split}.json", "w") as hundle:
                json.dump(self.videos, hundle, indent=2)

        self.max_length = max_length
        if showsize:
            _, counts = np.unique([f"{x[1]}{x[2]}" for x in self.videos], return_counts=True)
            print(split)
            print(f"Real Video Real Audio: {counts[0]}")
            print(f"Real Video Fake Audio: {counts[1]}")
            print(f"Fake Video Real Audio: {counts[2]}")
            print(f"Fake Video Fake Audio: {counts[3]}\n")

    def __len__(self):
        return len(self.videos)

    def __getmaxlength__(self):
        max_length = 0
        durations = []
        for idx in range(self.__len__()):
            try:
                data = np.load(self.videos[idx][0], allow_pickle=True)
                video_features = torch.tensor(data["video_features"])
                audio_features = torch.tensor(data["audio_features"])
                t = min(video_features.shape[0], audio_features.shape[0])
                durations.append(t)
                max_length = max(max_length, t)
            except:
                pass
        print("25th perc.", np.percentile(durations, 25))
        print("50th perc.", np.percentile(durations, 50))
        print("75th perc.", np.percentile(durations, 75))
        print("90th perc.", np.percentile(durations, 90))
        print("95th perc.", np.percentile(durations, 95))
        print("99th perc.", np.percentile(durations, 99))
        return max_length

    def period2target(self, fake_periods):
        tfl_target = torch.zeros((self.max_length, 3))
        for fake_period in fake_periods:
            fake_period_indx = [int(x * 25) for x in fake_period]
            tfl_target[fake_period_indx[0] : fake_period_indx[1] + 1, 0] = 1
            for i in range(fake_period_indx[0], fake_period_indx[1] + 1):
                tfl_target[i, 1] = i - fake_period_indx[0]
                tfl_target[i, 2] = fake_period_indx[1] - i
        return tfl_target

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            data_path, video_target, audio_target, fake_periods = self.videos[idx]
            if self.backbone == "avhubert":
                data_path = data_path.replace("data/LAV-DF_emb", "data/LAV-DF_emb_avhubert").replace("mediapipe/", "")
            data = np.load(data_path, allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat(
                [video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])]
            )
            audio_features = torch.concat(
                [audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])]
            )
            if self.task == "tfl":
                tfl_target = self.period2target(fake_periods)
                if video_target == 0 and audio_target == 0:
                    tfl_target = torch.zeros((self.max_length * 2, 3))
                elif video_target == 1 and audio_target == 0:
                    tfl_target = torch.cat((tfl_target, torch.zeros((self.max_length, 3))), dim=0)
                elif video_target == 0 and audio_target == 1:
                    tfl_target = torch.cat((torch.zeros((self.max_length, 3)), tfl_target), dim=0)
                elif video_target == 1 and audio_target == 1:
                    tfl_target = torch.cat((tfl_target, tfl_target), dim=0)

                return [video_features, audio_features, tfl_target, fake_periods]
            else:
                return [video_features, audio_features, video_target, audio_target]

        except:
            return None


class AVDeepFake1M(Dataset):

    def __init__(self, task, split, max_length, partition="partial", showsize=True):
        assert partition in ["partial", "whole"]
        self.name = "avdeepfake1m"
        self.num_partial_samples = 200000
        if split == "test":
            split = "val"
        self.task = task
        if partition == "partial" or split != "train":
            self.create_video_dict(filename=f"utils/avdeepfake1m_{split}.json", split=split, partition=partition)
        else:
            self.create_video_dict(filename=f"utils/avdeepfake1m_{split}_whole.json", split=split, partition=partition)

        self.max_length = max_length
        if showsize:
            _, counts = np.unique([f"{x[1]}{x[2]}" for x in self.videos], return_counts=True)
            print(split)
            print(f"Real Video Real Audio: {counts[0]}")
            print(f"Real Video Fake Audio: {counts[1]}")
            print(f"Fake Video Real Audio: {counts[2]}")
            print(f"Fake Video Fake Audio: {counts[3]}\n")

    def __len__(self):
        return len(self.videos)

    def create_video_dict(self, filename, split, partition):
        if os.path.exists(filename):
            self.read_json_file(filename)
        else:
            self.videos = self.get_video_paths(split, partition)
            self.write_json_file(filename, self.videos)

    def read_json_file(self, filename):
        with open(filename, "r") as hundle:
            self.videos = json.load(hundle)

    def write_json_file(self, filename, variable):
        with open(filename, "w") as hundle:
            json.dump(variable, hundle, indent=2)

    def get_video_paths(self, split, partition):
        directory = f"data/AV-Deepfake1M_emb/{split}"
        metadata = self.get_metadata(split)
        self.videos = []
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                if name == "features.npz":
                    fn = "/".join(os.path.dirname(root).split("/")[-4:]) + ".mp4"
                    targets = metadata[fn]
                    self.videos.append(
                        (os.path.join(root, name), targets[0], targets[1], targets[2], targets[3], targets[4])
                    )
            if (split == "train") and (partition == "partial") and (len(self.videos) == self.num_partial_samples):
                break
        return self.videos

    def get_metadata(self, split):
        with open(f"data/AV-Deepfake1M_emb/{split}_metadata.json", "r") as f:
            metadata = {
                x["file"]: (
                    (0, 0, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                    if x["modify_type"] == "real"
                    else (
                        (1, 1, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                        if x["modify_type"] == "both_modified"
                        else (
                            (1, 0, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                            if x["modify_type"] == "visual_modified"
                            else (0, 1, x["visual_fake_segments"], x["audio_fake_segments"], x["fake_segments"])
                        )
                    )
                )
                for x in json.load(f)
            }
        return metadata

    def __getmaxlength__(self):
        max_length = 0
        durations = []
        for idx in range(self.__len__()):
            try:
                data = np.load(self.videos[idx][0], allow_pickle=True)
                video_features = torch.tensor(data["video_features"])
                audio_features = torch.tensor(data["audio_features"])
                t = min(video_features.shape[0], audio_features.shape[0])
                durations.append(t)
                max_length = max(max_length, t)
            except:
                pass
        print("25th perc.", np.percentile(durations, 25))
        print("50th perc.", np.percentile(durations, 50))
        print("75th perc.", np.percentile(durations, 75))
        print("90th perc.", np.percentile(durations, 90))
        print("95th perc.", np.percentile(durations, 95))
        print("99th perc.", np.percentile(durations, 99))
        return max_length

    def period2target(self, fake_segments):
        tfl_target = torch.zeros((self.max_length, 3))
        for fake_period in fake_segments:
            fake_period_indx = [int(x * 25) for x in fake_period]
            tfl_target[fake_period_indx[0] : fake_period_indx[1] + 1, 0] = 1
            for i in range(fake_period_indx[0], fake_period_indx[1] + 1):
                tfl_target[i, 1] = i - fake_period_indx[0]
                tfl_target[i, 2] = fake_period_indx[1] - i
        return tfl_target

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            data_path, video_target, audio_target, visual_fake_segments, audio_fake_segments, fake_periods = (
                self.videos[idx]
            )
            data = np.load(data_path, allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat(
                [video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])]
            )
            audio_features = torch.concat(
                [audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])]
            )
            if self.task == "tfl":
                tfl_v_target = self.period2target(visual_fake_segments)
                tfl_a_target = self.period2target(audio_fake_segments)
                tfl_target = torch.cat((tfl_v_target, tfl_a_target), dim=0)
                return [video_features, audio_features, tfl_target, fake_periods]
            else:
                return [video_features, audio_features, video_target, audio_target]
        except:
            return None


class DFDCTest(Dataset):

    def __init__(self, detector, max_length, split="test", showsize=True):
        metadata = self.get_metadata(split=split)
        self.videos = []
        for root, dirs, files in os.walk(f"data/DFDC_emb/{split}", topdown=False):
            if detector in root:
                for name in files:
                    if name == "features.npz":
                        video_name = f'{root.split("/")[-2]}.mp4'
                        label = metadata[video_name]["label"]
                        self.videos.append((os.path.join(root, name), label))
        self.max_length = max_length

        if showsize:
            print(f"{split} set:")
            negative = sum([x[1] == 0 for x in self.videos])
            positive = sum([x[1] == 1 for x in self.videos])
            print(f"Real: {negative:,}")
            print(f"Fake: {positive:,}\n")

    def get_metadata(self, split):
        if split == "train":
            metadata = {}
            for i in range(50):
                with open(f"data/DFDC_emb/train/dfdc_train_part_{i}/metadata.json") as f:
                    metadata.update(json.load(f))
            metadata = {x: {"label": int(metadata[x]["label"] == "FAKE")} for x in metadata}
        elif split in ["validation", "test"]:
            metadata = pd.read_csv(f"data/DFDC_emb/{split}/labels.csv", index_col=0).to_dict("index")
        else:
            raise Exception(f"{split} split not supported")
        return metadata

    def __len__(self):
        return len(self.videos)

    def __getmaxlength__(self):
        max_length = 0
        durations = []
        for idx in range(self.__len__()):
            try:
                data = np.load(self.videos[idx][0], allow_pickle=True)
                video_features = torch.tensor(data["video_features"])
                audio_features = torch.tensor(data["audio_features"])
                t = min(video_features.shape[0], audio_features.shape[0])
                durations.append(t)
                max_length = max(max_length, t)
            except:
                pass
        print("25th perc.", np.percentile(durations, 25))
        print("50th perc.", np.percentile(durations, 50))
        print("75th perc.", np.percentile(durations, 75))
        print("90th perc.", np.percentile(durations, 90))
        print("95th perc.", np.percentile(durations, 95))
        print("99th perc.", np.percentile(durations, 99))
        return max_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path, target = self.videos[idx]
        try:
            data = np.load(data_path, allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat(
                [
                    video_features[:t, :],
                    torch.zeros([self.max_length - t, video_features.shape[1]]),
                ]
            )
            audio_features = torch.concat(
                [
                    audio_features[:t, :],
                    torch.zeros([self.max_length - t, audio_features.shape[1]]),
                ]
            )

            return [video_features, audio_features, target, target]
        except:
            return None


class KoDFTest(Dataset):

    def __init__(self, max_length, showsize=True):
        self.videos = []
        for root, dirs, files in os.walk(f"data/KoDF_emb", topdown=False):
            for name in files:
                if name == "features.npz":
                    label = 1 if "synthesized_videos" in root else 0
                    self.videos.append((os.path.join(root, name), label))

        self.max_length = max_length

        if showsize:
            print("Test set:")
            negative = sum([x[1] == 0 for x in self.videos])
            positive = sum([x[1] == 1 for x in self.videos])
            print(f"Real: {negative:,}")
            print(f"Fake: {positive:,}\n")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path, target = self.videos[idx]
        try:
            data = np.load(data_path, allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat(
                [
                    video_features[:t, :],
                    torch.zeros([self.max_length - t, video_features.shape[1]]),
                ]
            )
            audio_features = torch.concat(
                [
                    audio_features[:t, :],
                    torch.zeros([self.max_length - t, audio_features.shape[1]]),
                ]
            )

            return [video_features, audio_features, target, target]
        except:
            return None
