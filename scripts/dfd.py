from src.training import Experiment
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


parser = argparse.ArgumentParser(description="Deepfake detection experiments")
parser.add_argument(
    "-d",
    "--dataset_name",
    help="the name of the dataset to conduct experiments on",
    choices=["fakeavceleb", "lavdf", "avdeepfake1m"],
)
parser.add_argument(
    "-f",
    "--feature_pyramid",
    help="whether to use feature pyramids",
    default="True",
    choices=["True", "False"],
)
parser.add_argument(
    "-g",
    "--gpu",
    help="which gpu to use",
    default="0",
)
parser.add_argument(
    "-p",
    "--hparams",
    help="d_model, nhead, and multiplier hyperparam values",
    default="[[32,2,1],[64,2,1],[64,4,1],[128,4,2],[256,8,2]]",
)
parser.add_argument(
    "-l",
    "--layers",
    help="number of layers",
    default="[1,3,5]",
)
parser.add_argument(
    "-s",
    "--schedulers",
    help="schedulers to use",
    default="none,reduceonplateau,step,cosineanealing",
)
parser.add_argument(
    "-i",
    "--id",
    help="slurm array task id",
)
args = parser.parse_args()

task = "dfd"
max_length = 600
disable_tqdm = False
dataset = args.dataset_name
if args.feature_pyramid == "True":
    feature_pyramid = True
elif args.feature_pyramid == "False":
    feature_pyramid = False
else:
    raise Exception("feature pyramyd value should be either True or False")
gpu_index = args.gpu
schedulers = args.schedulers.split(",")
params = [
    {
        "d_model": d_model,
        "nhead": nhead,
        "multiplier": multiplier,
        "nlayers": nlayers,
        "scheduler": scheduler,
    }
    for d_model, nhead, multiplier in eval(args.hparams)
    for nlayers in eval(args.layers)
    for scheduler in schedulers
]
experiment_id = int(args.id)
experiment_setup = params[experiment_id]

win_size = 15
batch_size = 64
lr = 0.001
epochs = 100
workers = 22
seed = 0
patience = 10
device = f"cuda:{gpu_index}"
logging = True
alpha = None
folder = os.path.join("results", task)

filename = "_".join(
    map(
        str,
        [
            task,
            dataset,
            experiment_setup["scheduler"],
            experiment_setup["d_model"],
            experiment_setup["nhead"],
            experiment_setup["multiplier"],
            experiment_setup["nlayers"],
            feature_pyramid,
        ],
    )
)
filepath = os.path.join(folder, f"{filename}.json")
print(filepath)
if not check_complete(filepath):
    experiment = Experiment(
        dataset=dataset,
        task=task,
        alpha=alpha,
        max_length=max_length,
        feature_pyramid=feature_pyramid,
        d_model=experiment_setup["d_model"],
        nhead=experiment_setup["nhead"],
        d_hid=experiment_setup["multiplier"] * experiment_setup["d_model"],
        nlayers=experiment_setup["nlayers"],
        win_size=win_size,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        workers=workers,
        seed=seed,
        folder=folder,
        filename=filename,
        patience=patience,
        logging=logging,
        scheduler=experiment_setup["scheduler"],
        disable_tqdm=disable_tqdm,
        delete_ckpt=True,
        device=device,
    )
    experiment.run()
