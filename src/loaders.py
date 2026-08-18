import torch
from torch.utils.data import DataLoader
from src.datasets import FakeAVCeleb, LAVDF, AVDeepFake1M, DFDCTest, KoDFTest
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn_dfd(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def collate_fn_tfl(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch_1 = [b[:3] for b in batch]
    batch_2 = [b[3] for b in batch]
    return torch.utils.data.dataloader.default_collate(batch_1), batch_2


def get_dataset(task, dataset, partition, split, max_length, without, backbone, showsize):
    if dataset == "fakeavceleb":
        assert task == "dfd"
        return FakeAVCeleb(split=split, max_length=max_length, without=without, backbone=backbone, showsize=showsize)
    elif dataset == "lavdf":
        return LAVDF(task=task, split=split, max_length=max_length, backbone=backbone, showsize=showsize)
    elif dataset == "avdeepfake1m":
        return AVDeepFake1M(task=task, split=split, max_length=max_length, partition=partition, showsize=showsize)
    elif dataset == "dfdc":
        return DFDCTest(detector="mediapipe", max_length=max_length, split="test", showsize=showsize)
    elif dataset == "kodf":
        return KoDFTest(max_length=max_length, showsize=showsize)
    else:
        raise Exception(f"Dataset {dataset} is not supported.")


def get_loaders(
    task,
    dataset,
    partition,
    max_length,
    batch_size,
    workers,
    without="none",
    backbone="ma22",
    splits=None,
    showsize=True,
):
    if splits is None:
        splits = ["train", "val", "test"]

    loaders = {}
    for split in splits:
        loaders[split] = DataLoader(
            get_dataset(task, dataset, partition, split, max_length, without, backbone, showsize),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn_tfl if task == "tfl" else collate_fn_dfd,
            pin_memory=True,
            drop_last=False,
        )
    return loaders
