import torch

from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

import os
import datetime
import json

from src.loaders import get_loaders
from src.models import AVDeepFakeDetector
from src.eval import Evaluation, adjust_data
from src.logger import Logger
from src.losses import FocalLoss, CombinedLoss, BCELoss
from src.seed import seed_everything


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Experiment:

    def __init__(
        self,
        dataset,
        task,
        alpha,
        max_length,
        feature_pyramid,
        d_model,
        nhead,
        d_hid,
        nlayers,
        win_size,
        batch_size,
        lr,
        epochs,
        workers,
        seed,
        folder,
        filename,
        patience,
        logging,
        scheduler,
        disable_tqdm,
        delete_ckpt,
        device,
    ):
        self.dataset = dataset
        self.task = task
        self.alpha = alpha
        self.max_length = max_length
        self.feature_pyramid = feature_pyramid
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.win_size = win_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.workers = workers
        self.seed = seed
        self.folder = folder
        self.filename = filename
        self.ckpt_folder = "/".join(["ckpt"] + folder.split(os.sep)[1:])
        if not os.path.exists(self.ckpt_folder):
            os.makedirs(self.ckpt_folder)
        self.ckpt_path = "/".join([self.ckpt_folder] + [f"{filename}.pth"])
        self.patience = patience
        self.logging = logging
        self.scheduler = scheduler
        self.disable_tqdm = disable_tqdm
        self.delete_ckpt = delete_ckpt
        self.device = device
        self.configuration = {
            "dataset": self.dataset,
            "task": self.task,
            "criterion_name": "combined" if self.task == "tfl" else "bce",
            "alpha": self.alpha,
            "max_length": self.max_length,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "d_hid": self.d_hid,
            "nlayers": self.nlayers,
            "win_size": self.win_size,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "seed": self.seed,
            "patience": self.patience,
            "scheduler": self.scheduler,
            "feature_pyramid": self.feature_pyramid,
        }
        self.partition = "partial"
        self.without = "none"
        self.backbone = "ma22"

    def set_attribute(self, attribute, value):
        print(f"Set {attribute} to {value}")
        exec(f"self.{attribute}={value}")
        self.configuration[attribute] = value

    def get_model(self):
        return AVDeepFakeDetector(
            task=self.task,
            max_length=self.max_length,
            d_model=self.d_model,
            nhead=self.nhead,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            win_size=self.win_size,
            feature_pyramid=self.feature_pyramid,
            device=self.device,
        )

    def get_scheduler(self, name, optimizer):
        if name == "none":
            return None
        elif name == "reduceonplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=self.patience - 3)
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, self.epochs // 5)
        elif name == "cosineanealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

    def get_criterion(self, name, gamma=2, reduction="mean"):
        if name == "bce":
            return BCELoss()
        elif name == "focal":
            return FocalLoss(alpha=self.alpha, gamma=gamma, reduction=reduction)
        elif name == "combined":
            return CombinedLoss(alpha=self.alpha, gamma=gamma)
        else:
            raise Exception(f"Criterion {name} is not supported.")

    def compute_metric(self, labels, predictions):
        if self.task == "dfd":
            if self.model.feature_pyramid:
                return accuracy_score(
                    labels.unsqueeze(1).repeat(1, predictions.shape[1], 1).reshape(-1, 2).cpu().numpy(),
                    torch.sigmoid(predictions.reshape(-1, 2)).detach().cpu().numpy() > 0.5,
                )
            else:
                return accuracy_score(
                    labels.cpu().numpy(),
                    torch.sigmoid(predictions).detach().cpu().numpy() > 0.5,
                )
        else:
            if self.model.feature_pyramid:
                return average_precision_score(
                    labels[:, :, 0]
                    .unsqueeze(1)
                    .repeat(1, predictions.shape[1], 1)
                    .reshape(-1, predictions.shape[-1])
                    .cpu()
                    .numpy(),
                    torch.sigmoid(predictions[:, :, :, 0].reshape(-1, predictions.shape[-1])).detach().cpu().numpy(),
                    average="micro",
                )
            else:
                return average_precision_score(
                    labels[:, :, 0].cpu().numpy(),
                    torch.sigmoid(predictions[:, :, 0]).detach().cpu().numpy(),
                    average="micro",
                )

    def adjust_metrics(self, loss, metric_name, metric, validation):
        if self.task == "tfl":
            l = {"loss": validation["loss"]}
            m = {f"{x}@{y}": validation[x][y] for x in ["ap", "ar"] for y in validation[x]}
            validation = {**l, **m}
            pf_keys = ["vloss", "vap@0.5"]
        else:
            pf_keys = ["vloss", "vauc"]
        t = {"loss": loss, metric_name: 100 * metric}
        v = {f"v{x}": (99 * int(x != "loss") + 1) * validation[x] for x in validation}
        metrics = {**t, **v}
        postfix = {x: v[x] for x in pf_keys}
        postfix = {**t, **postfix}
        return metrics, postfix

    def train_one_epoch(self, epoch):
        self.model.train()
        tot_loss, tot_metric = 0, 0
        metric_name = "acc" if self.task == "dfd" else "ap"
        with tqdm(
            unit="batch", total=len(self.loaders["train"]), dynamic_ncols=True, disable=self.disable_tqdm
        ) as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for iteration, data in enumerate(self.loaders["train"]):
                tepoch.update(1)
                video_features, audio_features, labels, _ = adjust_data(self.task, data, self.device)
                p, z = self.model([video_features, audio_features])
                self.optimizer.zero_grad()
                loss_ = self.criterion(p, labels)
                tot_loss += loss_.item()
                loss_.backward()
                self.optimizer.step()
                tot_metric += self.compute_metric(labels, p)
                tepoch.set_postfix(
                    {"loss": tot_loss / (iteration + 1), metric_name: 100.0 * tot_metric / (iteration + 1)}
                )

            e = Evaluation(
                task=self.task,
                model=self.model,
                loader=self.loaders["val"],
                criterion=self.criterion,
                feature_pyramid=self.model.feature_pyramid,
                device=self.device,
            )
            e.compute_metrics()
            metrics, postfix = self.adjust_metrics(
                loss=tot_loss / len(self.loaders["train"]),
                metric_name=metric_name,
                metric=tot_metric / len(self.loaders["train"]),
                validation=e.metrics,
            )
            if self.scheduler_dict["obj"] is not None:
                if self.scheduler_dict["name"] == "reduceonplateau":
                    if self.task == "dfd":
                        current_score = metrics["vauc"]
                    else:
                        current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
                    self.scheduler_dict["obj"].step(current_score)
                else:
                    self.scheduler_dict["obj"].step()
            tepoch.set_postfix(postfix)
            tepoch.close()
            return metrics

    def training_process(self):
        best_score = 0
        best_epoch = 0
        metrics_train_val = []
        for epoch in range(self.epochs):
            start_epoch = datetime.datetime.now()
            metrics = self.train_one_epoch(epoch)
            metrics_train_val.append(metrics)
            if self.disable_tqdm:
                duration = datetime.datetime.now() - start_epoch
                print(f"[Epoch {epoch+1}/{self.epochs} {str(duration)}]", metrics)
            self.logger.update("training", metrics_train_val)
            if self.task == "dfd":
                current_score = metrics["vauc"]
            else:
                current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                torch.save({"optimizer": self.optimizer.state_dict(), "model": self.model.state_dict()}, self.ckpt_path)
                print(f"Saved checkpoint at {self.ckpt_path}")
            elif epoch - best_epoch >= self.patience:
                print(f"Early stopped training at epoch {epoch+1}")
                break

        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint["model"])
        e = Evaluation(
            task=self.task,
            model=self.model,
            loader=self.loaders["test"],
            criterion=self.criterion,
            feature_pyramid=self.model.feature_pyramid,
            device=self.device,
        )
        e.compute_metrics()
        if self.task == "dfd":
            metrics_test = {f"t{x}": (99 * int(x != "loss") + 1) * e.metrics[x] for x in e.metrics}
        else:
            l = {"tloss": e.metrics["loss"]}
            m = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
            metrics_test = {**l, **m}

        print(json.dumps(metrics_test, indent=2))
        self.logger.update("test", metrics_test)
        if self.delete_ckpt:
            os.remove(self.ckpt_path)

    def run(self):
        start_time = datetime.datetime.now()
        seed_everything(self.seed)

        self.logger = Logger(folder=self.folder, filename=self.filename, enable=self.logging)
        self.logger.create()
        self.logger.update("config", self.configuration)

        self.loaders = get_loaders(
            task=self.task,
            dataset=self.dataset,
            partition=self.partition,
            max_length=self.max_length,
            batch_size=self.batch_size,
            workers=self.workers,
            without=self.without,
            backbone=self.backbone,
        )
        self.model = self.get_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler_dict = {"name": self.scheduler, "obj": self.get_scheduler(self.scheduler, self.optimizer)}
        self.criterion = self.get_criterion(name=self.configuration["criterion_name"])
        self.training_process()

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        self.logger.update("time", str(duration))
        print(f"time: {str(duration)}")
