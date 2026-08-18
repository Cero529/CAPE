from src.models import AVDeepFakeDetector
from src.post_process import soft_nms_torch_parallel
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import json
import random
import argparse

torch.backends.mha.set_fastpath_enabled(False)


class AVDeepFake1MTestSet(Dataset):

    def __init__(self, max_length, showsize=True):
        self.videos = [f"data/AV-Deepfake1M_emb/test/{i:06d}/mediapipe/features.npz" for i in range(343240)]

        self.max_length = max_length
        if showsize:
            print(f"Test set: {self.__len__():,}\n")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_id = f'{self.videos[idx].split("/")[3]}.mp4'
        try:
            data = np.load(self.videos[idx], allow_pickle=True)
            video_features = torch.tensor(data["video_features"])
            audio_features = torch.tensor(data["audio_features"])
            t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
            video_features = torch.concat(
                [video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])]
            )
            audio_features = torch.concat(
                [audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])]
            )
            return [video_features, audio_features, video_id]
        except:
            return [torch.zeros([self.max_length, 768]), torch.zeros([self.max_length, 768]), video_id]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def transform_tfl_predictions(preds, max_length, device):
    sigma, t1, t2, fps = 0.7234, 0.1968, 0.4123, 25
    idx_ = torch.arange(0, max_length).to(device)
    idx = torch.cat((idx_, idx_)).unsqueeze(0)
    idx = idx.repeat(1, preds.shape[1])
    preds = preds.reshape(preds.shape[0], -1, 3)
    preds[:, :, 0] = torch.sigmoid(preds[:, :, 0])
    preds[:, :, 1] = torch.clamp(idx - preds[:, :, 1], min=0.0)
    preds[:, :, 2] = torch.clamp(idx + preds[:, :, 2], max=max_length)
    _, indexes = torch.sort(preds[:, :, 0], dim=1, descending=True)
    first_indices = torch.arange(preds.shape[0])[:, None]
    preds = preds[first_indices, indexes]
    proposals = soft_nms_torch_parallel(preds, sigma, t1, t2, fps, device)
    return proposals


def get_stored_predictions(task, path):
    if task == "tfl":
        if os.path.exists(path):
            with open(path, "r") as f:
                processed_files = [list(json.loads(line).keys())[0] for line in f]
        else:
            processed_files = []
    elif task == "dfd":
        if os.path.exists(path):
            with open(path, "r") as f:
                processed_files = [line.split(";")[0] for line in f]
        else:
            processed_files = []
    else:
        raise Exception(f"{task} task not supported")

    return processed_files


parser = argparse.ArgumentParser(description="AV-Deepfake1M test set predictions")
parser.add_argument(
    "-t",
    "--task",
    help="name of the task",
    choices=["dfd", "tfl"],
)
parser.add_argument(
    "-p",
    "--partition",
    help="name of the partition",
    default="partial",
    choices=["partial", "whole"],
)
args = parser.parse_args()
task = args.task
partition = args.partition
device = "cuda:0"
metrics_device = "cpu"
model_files = {
    "dfd": {
        "partial": "ckpt/dfd/dfd_avdeepfake1m_reduceonplateau_64_4_1_5_True.json",
        "whole": "ckpt/dfd/dfd_avdeepfake1m_reduceonplateau_64_4_1_5_True_whole.json",
    },
    "tfl": {
        "partial": "ckpt/tfl/tfl_avdeepfake1m_reduceonplateau_256_8_2_5_True.json",
        "whole": "ckpt/tfl/tfl_avdeepfake1m_reduceonplateau_256_8_2_5_True_whole.json",
    },
}
max_length = 600
fps = 25
batch_size = 64
workers = 4
dataset = AVDeepFake1MTestSet(max_length=max_length)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    worker_init_fn=seed_worker,
    pin_memory=True,
    drop_last=False,
)

model_json_file = model_files[task][partition]
model_ckpt_file = f'{model_json_file.split(".")[0]}.pth'
with open(model_json_file, "r") as hundle:
    configuration = json.load(hundle)["config"]
model = AVDeepFakeDetector(
    task=task,
    max_length=max_length,
    d_model=configuration["d_model"],
    nhead=configuration["nhead"],
    d_hid=configuration["d_hid"],
    nlayers=configuration["nlayers"],
    win_size=configuration["win_size"],
    feature_pyramid=bool(configuration["feature_pyramid"]),
    device=device,
)
model.to(device)
checkpoint = torch.load(model_ckpt_file)
model.load_state_dict(checkpoint["model"])
model.eval()

folder = f"utils/avdeepfake1m_test_predictions/{task}_{partition}"
if not os.path.exists(folder):
    os.makedirs(folder)
filename = f"{folder}/prediction"
extention = "txt" if task == "dfd" else "jsonl"
filename_ext = f"{filename}.{extention}"

if task == "tfl":
    processed_video_ids = get_stored_predictions(task, filename_ext)
    for data in loader:
        video_features, audio_features, video_ids = data
        if all([v in processed_video_ids for v in video_ids]):
            continue
        video_features, audio_features = video_features.to(device), audio_features.to(device)
        with torch.no_grad():
            p, z = model([video_features, audio_features])
        proposals = (
            transform_tfl_predictions(p.detach().cpu(), max_length, metrics_device).detach().cpu().numpy().tolist()
        )
        predictions = {
            x: [[y_0 / fps if i != 0 else y_0 for i, y_0 in enumerate(y_)] for y_ in y]
            for x, y in zip(video_ids, proposals)
        }
        with open(filename_ext, "a") as hundle:
            for x in predictions:
                json.dump({x: predictions[x]}, hundle)
                hundle.write("\n")
elif task == "dfd":
    processed_video_ids = get_stored_predictions(task, filename_ext)
    for data in loader:
        video_features, audio_features, video_ids = data
        if all([v in processed_video_ids for v in video_ids]):
            continue
        video_features, audio_features = video_features.to(device), audio_features.to(device)
        with torch.no_grad():
            p, z = model([video_features, audio_features])
        predictions = torch.sigmoid(p.mean(dim=1)).max(dim=-1)[0].detach().cpu().numpy().tolist()
        with open(filename_ext, "a" if os.path.exists(filename_ext) else "w") as hundle:
            hundle.writelines([f"{x};{y:1.4f}\n" for x, y in zip(video_ids, predictions)])
else:
    raise Exception(f"{task} task not supported")
