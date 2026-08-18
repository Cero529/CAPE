from src.training import Experiment
from scripts.results import get_best
import os
import argparse
import json


def check_complete(path):
    if os.path.exists(path):
        with open(path, "r") as hundle:
            json_file = json.load(hundle)
            if "test" in json_file:
                return True
            else:
                return False
    else:
        return False


parser = argparse.ArgumentParser(description="Ablations")
parser.add_argument(
    "-a",
    "--ablation",
    help="ablation",
    default="win_size",
    choices=["win_size", "alpha", "feature_pyramid"],
)
parser.add_argument(
    "-g",
    "--gpu",
    help="which gpu to use",
    default="0",
)
parser.add_argument(
    "-i",
    "--id",
    help="slurm array task id",
)
args = parser.parse_args()

gpu_index = args.gpu
ablation = args.ablation
experiment_id = int(args.id)

disable_tqdm = True
workers = 22
device = f"cuda:{gpu_index}"
logging = True
folder = os.path.join("results", "ablation")
datasets = ["fakeavceleb", "lavdf", "avdeepfake1m"] if ablation != "alpha" else ["lavdf", "avdeepfake1m"]
tasks = ["dfd", "tfl"] if ablation != "alpha" else ["tfl"]
values = {
    "win_size": [5, 51, 0],
    "alpha": [0.2, 0.7, 0.9],
    "feature_pyramid": [False],
}
experiment_setups = [
    {"dataset": ds, "task": tk, "value": vl}
    for ds in datasets
    for tk in tasks
    for vl in values[ablation]
    if not (ds == "fakeavceleb" and tk == "tfl")
]
experiment_setup = experiment_setups[experiment_id]
filename, configuration, training, performance = get_best(
    task=experiment_setup["task"], term=f"{experiment_setup['dataset']}_reduceonplateau"
)
filename = f"{filename}_{ablation}_{experiment_setup['value']}"
filepath = os.path.join(folder, f"{filename}.json")
print(filepath)
if not check_complete(filepath):
    experiment = Experiment(
        dataset=experiment_setup["dataset"],
        task=experiment_setup["task"],
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
        delete_ckpt=True,
        device=device,
    )
    experiment.set_attribute(ablation, experiment_setup["value"])
    print(experiment.configuration)
    experiment.run()
