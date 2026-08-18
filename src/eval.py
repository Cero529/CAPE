from src.metrics import AP, AR
from src.post_process import soft_nms_torch_parallel

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import numpy as np

import torch


def adjust_data(task, data, device):
    if task == "dfd":
        video_features, audio_features, video_targets, audio_targets = data
        labels = torch.cat((video_targets.unsqueeze(1), audio_targets.unsqueeze(1)), dim=1)
        fake_periods = None
    elif task == "tfl":
        data, fake_periods = data
        video_features, audio_features, labels = data
    else:
        raise Exception(f"{task} task not supported")
    video_features, audio_features, labels = (
        video_features.to(device),
        audio_features.to(device),
        labels.float().to(device),
    )
    return video_features, audio_features, labels, fake_periods


class Evaluation:

    def __init__(self, task, model, loader, criterion, feature_pyramid, device, generalization=False):
        self.task = task
        self.model = model
        self.loader = loader
        self.max_length = loader.dataset.max_length
        self.dataset = "generalization" if generalization else loader.dataset.name
        self.criterion = criterion
        self.feature_pyramid = feature_pyramid
        self.device = device
        self.metrics_device = "cpu"
        self.loss = 0
        self.ground_truth = []
        self.predictions = {"dfd": [], "tfl": None}
        self.update_predictions = {"dfd": self.update_predictions_dfd, "tfl": self.update_predictions_tfl}
        self.sigma, self.t1, self.t2, self.fps = 0.7234, 0.1968, 0.4123, 25
        self.confidence_thres = 0.5
        self.n_proposals_list = {
            "lavdf": [100, 50, 20, 10],
            "avdeepfake1m": [50, 30, 20, 10, 5],
            "generalization": [100, 50, 30, 20, 10, 5],
        }
        self.ar_iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.ap_iou_thresholds = {
            "lavdf": [0.5, 0.75, 0.95],
            "avdeepfake1m": [0.5, 0.75, 0.9, 0.95],
            "generalization": [0.5, 0.75, 0.9, 0.95],
        }
        self.metrics = {}

    def get_predictions(self):
        self.model.eval()
        with torch.no_grad():
            for data in self.loader:
                video_features, audio_features, labels, fake_periods_ = adjust_data(self.task, data, self.device)
                p, z = self.model([video_features, audio_features])
                self.loss += self.criterion(p, labels).item()
                self.update_predictions[self.task](p, labels if self.task == "dfd" else fake_periods_)
        self.loss /= len(self.loader)

    def compute_metrics(self):
        self.get_predictions()
        if self.task == "dfd":
            if self.feature_pyramid:
                pr_ = np.array(self.predictions[self.task]).mean(axis=1).max(axis=-1)
            else:
                pr_ = np.array(self.predictions[self.task]).max(axis=-1)
            gt_ = torch.tensor(self.ground_truth).cpu().numpy().max(axis=-1)
            acc = accuracy_score(gt_, pr_ > 0.5)
            ap = average_precision_score(gt_, pr_)
            auc = roc_auc_score(gt_, pr_)
            self.metrics = {"loss": self.loss, "acc": acc, "ap": ap, "auc": auc}
        else:
            self.transform_predictions()
            proposals = soft_nms_torch_parallel(
                self.predictions[self.task], self.sigma, self.t1, self.t2, self.fps, self.metrics_device
            )
            ap = AP(
                iou_thresholds=self.ap_iou_thresholds[self.dataset],
                device=self.metrics_device,
            )
            ap_score = ap(proposals, self.ground_truth)
            ar = AR(
                n_proposals_list=self.n_proposals_list[self.dataset],
                iou_thresholds=self.ar_iou_thresholds,
                device=self.metrics_device,
            )
            ar_score = ar(proposals, self.ground_truth)
            self.metrics = {"loss": self.loss, "ap": ap_score, "ar": ar_score}

    def update_predictions_dfd(self, predictions, labels):
        self.ground_truth.extend(labels.cpu().numpy().tolist())
        self.predictions[self.task].extend(torch.sigmoid(predictions).detach().cpu().numpy().tolist())

    def update_predictions_tfl(self, predictions, labels):
        self.ground_truth.extend(labels)
        if self.predictions[self.task] is None:
            self.predictions[self.task] = predictions.detach().cpu()
        else:
            self.predictions[self.task] = torch.cat((self.predictions[self.task], predictions.detach().cpu()), dim=0)

    def transform_predictions(self):
        idx_ = torch.arange(0, self.max_length).to(self.metrics_device)
        idx = torch.cat((idx_, idx_)).unsqueeze(0)
        if self.feature_pyramid:
            idx = idx.repeat(1, self.predictions[self.task].shape[1])
            self.predictions[self.task] = self.predictions[self.task].reshape(
                self.predictions[self.task].shape[0], -1, 3
            )
        self.predictions[self.task][:, :, 0] = torch.sigmoid(self.predictions[self.task][:, :, 0])
        self.predictions[self.task][:, :, 1] = torch.clamp(idx - self.predictions[self.task][:, :, 1], min=0.0)
        self.predictions[self.task][:, :, 2] = torch.clamp(
            idx + self.predictions[self.task][:, :, 2], max=self.max_length
        )
        _, indexes = torch.sort(self.predictions[self.task][:, :, 0], dim=1, descending=True)
        first_indices = torch.arange(self.predictions[self.task].shape[0])[:, None]
        self.predictions[self.task] = self.predictions[self.task][first_indices, indexes]
