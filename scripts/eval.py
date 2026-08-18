from src.training import Experiment
from src.eval import Evaluation
from src.loaders import get_loaders
from src.logger import Logger
from src.eval import Evaluation

import os
import json

import torch
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

torch.backends.mha.set_fastpath_enabled(False)


def get_json_file(task, dataset):
    if task == "dfd":
        if dataset == "fakeavceleb":
            return f"ckpt/dfd/dfd_fakeavceleb_reduceonplateau_64_4_1_3_True_none.json"
        elif dataset == "lavdf":
            return "ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json"
        elif dataset == "avdeepfake1m":
            return "ckpt/dfd/dfd_avdeepfake1m_reduceonplateau_64_4_1_5_True_whole.json"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    elif task == "tfl":
        if dataset == "lavdf":
            return "ckpt/tfl/tfl_lavdf_reduceonplateau_256_8_2_5_True.json"
        elif dataset == "avdeepfake1m":
            return "ckpt/tfl/tfl_avdeepfake1m_reduceonplateau_256_8_2_5_True_whole.json"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    else:
        raise ValueError(f"Unknown task: {task}")


def skip_check(path, task, train_ds, test_ds, test):
    if test:
        if test_ds in ["fakeavceleb", "dfdc", "kodf"] and task == "tfl":
            return True
        if os.path.exists(path):
            with open(path, "r") as hundle:
                json_file = json.load(hundle)
            if train_ds in json_file:
                evaluated_datasets = set(json_file[train_ds].keys())
                if test_ds in evaluated_datasets:
                    return True
    else:
        if train_ds == "fakeavceleb" and task == "tfl":
            return True
    return False


device = "cuda:0"
workers = 4
batch_size = 64
tasks = [
    "dfd",
    "tfl",
]
datasets = [
    "fakeavceleb",
    "lavdf",
    "avdeepfake1m",
]
test_datasets = [
    "fakeavceleb",
    "lavdf",
    "avdeepfake1m",
    "dfdc",
    "kodf",
]
withouts = [
    "rvfa",
    "fvra-wl",
    "fvfa-wl",
    "fvfa-fs",
    "fvfa-gan",
]
for task in tasks:
    for dataset in datasets:
        print(f"\nTask:{task} Dataset:{dataset}")

        logger = Logger(folder=f"results/generalization", filename=f"{task}_{dataset}", enable=True)

        if skip_check(path=logger.path, task=task, train_ds=dataset, test_ds="none", test=False):
            print("Skipping Training Configuration...")
            continue

        if not os.path.exists(logger.path):
            logger.create()

        json_file = get_json_file(task, dataset)
        with open(json_file, "r") as hundle:
            data = json.load(hundle)

        filename = json_file.split(".")[0]
        configuration = data["config"]

        test_loaders = []
        for test_dataset in test_datasets:
            if skip_check(path=logger.path, task=task, train_ds=dataset, test_ds=test_dataset, test=True):
                print(f"Skipping TestDataset:{test_dataset}...")
                continue
            test_loader = get_loaders(
                task=task,
                dataset=test_dataset,
                partition="whole",
                max_length=configuration["max_length"],
                batch_size=batch_size,
                workers=workers,
                splits=["test"],
                showsize=False,
            )
            test_loaders.append((test_dataset, test_loader["test"]))
        if task == "dfd":
            for without in withouts:
                test_loader = get_loaders(
                    task=task,
                    dataset="fakeavceleb",
                    partition="whole",
                    max_length=600,
                    batch_size=batch_size,
                    workers=workers,
                    without=without,
                    splits=["test"],
                    showsize=False,
                )
                test_loaders.append((f"fakeavceleb-wo-{without}", test_loader["test"]))

        if test_loaders:
            experiment = Experiment(
                dataset=dataset,
                task=task,
                alpha=configuration["alpha"],
                max_length=configuration["max_length"],
                feature_pyramid=configuration["feature_pyramid"],
                d_model=configuration["d_model"],
                nhead=configuration["nhead"],
                d_hid=configuration["d_hid"],
                nlayers=configuration["nlayers"],
                win_size=configuration["win_size"],
                batch_size=batch_size,
                lr=configuration["lr"],
                epochs=configuration["epochs"],
                workers=workers,
                seed=configuration["seed"],
                folder="none",
                filename="none",
                patience=configuration["patience"],
                logging=False,
                scheduler=configuration["scheduler"],
                disable_tqdm=True,
                delete_ckpt=True,
                device=device,
            )
            model = experiment.get_model()
            model.to(device)
            criterion = experiment.get_criterion(name=configuration["criterion_name"])

            for test_dataset, test_loader in test_loaders:
                wo_ = test_dataset[test_dataset.find("-wo-") + 4 :]
                ckpt_path = (
                    f"{filename.replace('_none', '')}_{wo_}.pth"
                    if ("-wo-" in test_dataset) and dataset == "fakeavceleb"
                    else f"{filename}.pth"
                )
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint["model"])
                e = Evaluation(
                    task=task,
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    feature_pyramid=configuration["feature_pyramid"],
                    device=device,
                    generalization=True,
                )
                if test_dataset in ["dfdc", "kodf"]:
                    e.get_predictions()
                    gt_ = np.array(e.ground_truth)[:, 0]
                    pr_ = np.array(e.predictions["dfd"]).mean(axis=1).max(axis=-1)
                    acc = accuracy_score(gt_, pr_ > 0.5)
                    ap = average_precision_score(gt_, pr_)
                    auc = roc_auc_score(gt_, pr_)
                    performance = {"tloss": e.loss, "tacc": acc * 100, "tap": ap * 100, "tauc": auc * 100}
                else:
                    e.compute_metrics()
                    if task == "dfd":
                        performance = {f"t{x}": (99 * int(x != "loss") + 1) * e.metrics[x] for x in e.metrics}
                    else:
                        l = {"tloss": e.metrics["loss"]}
                        m = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
                        performance = {**l, **m}
                logger.update([dataset, test_dataset], performance)
