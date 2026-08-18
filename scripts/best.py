from src.training import Experiment
from scripts.results import get_best
import os
import argparse


parser = argparse.ArgumentParser(description="Retrain best configurations")
parser.add_argument(
    "-d",
    "--dataset_name",
    help="the name of the dataset to conduct experiments on",
    choices=["fakeavceleb", "lavdf", "avdeepfake1m"],
)
parser.add_argument(
    "-t",
    "--task",
    help="name of the task",
    default="dfd",
    choices=["dfd", "tfl"],
)
parser.add_argument(
    "-p",
    "--partition",
    help="partial or whole training set",
    default="partial",
    choices=["partial", "whole"],
)
parser.add_argument(
    "-g",
    "--gpu",
    help="which gpu to use",
    default="0",
)
args = parser.parse_args()

dataset = args.dataset_name
task = args.task
gpu_index = args.gpu

disable_tqdm = True
workers = 22
device = f"cuda:{gpu_index}"
logging = True
folder = os.path.join("ckpt", task)
filename, configuration, training, performance = get_best(task=task, term=f"{dataset}_reduceonplateau")
if args.partition == "whole":
    assert dataset == "avdeepfake1m"
    filename = f"{filename}_{args.partition}"
filepath = os.path.join(folder, f"{filename}.json")
print(filepath)
print(configuration)
experiment = Experiment(
    dataset=dataset,
    task=task,
    alpha=configuration["alpha"],
    max_length=configuration["max_length"],
    feature_pyramid=bool(configuration["feature_pyramid"]),
    d_model=configuration["d_model"],
    nhead=configuration["nhead"],
    d_hid=configuration["d_hid"],
    nlayers=configuration["nlayers"],
    win_size=configuration["win_size"],
    batch_size=configuration["batch_size"],
    lr=configuration["lr"],
    epochs=configuration["epochs"],
    workers=workers,
    seed=configuration["seed"],
    folder=folder,
    filename=filename,
    patience=configuration["patience"],
    logging=logging,
    scheduler=configuration["scheduler"],
    disable_tqdm=disable_tqdm,
    delete_ckpt=False,
    device=device,
)
experiment.partition = args.partition
experiment.run()
