from src.training import Experiment
from scripts.results import get_best
import os
import argparse


parser = argparse.ArgumentParser(description="FakeAVCeleb in-dataset and cross-manipulation experiments")
parser.add_argument(
    "-i",
    "--index",
    help="index of without-fake-method, 0: none represents the in-dataset experiment, 1-5: represent the cross-manipulation experiments",
    default=0,
)
args = parser.parse_args()
method = ["none", "rvfa", "fvra-wl", "fvfa-fs", "fvfa-gan", "fvfa-wl"][int(args.index)]

dataset = "fakeavceleb"
task = "dfd"
disable_tqdm = True
workers = 8
device = f"cuda:0"
logging = True
folder = os.path.join("ckpt", task)
filename, configuration, training, performance = get_best(task=task, term=f"fakeavceleb_reduceonplateau")
filename = filename + f"_{method}"
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
experiment.without = method
experiment.configuration["without"] = method
experiment.run()
