import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from src.training import Experiment

torch.backends.mha.set_fastpath_enabled(False)


def get_gpu_hours():
    duration = pd.to_timedelta("0")
    for task in ["dfd", "tfl", "ablation"]:
        files = glob.glob(f"results/{task}/*")
        for file in files:
            with open(file, "r") as hundle:
                data_ = json.load(hundle)
                if "time" in data_:
                    duration += pd.to_timedelta(data_["time"])
    print(duration)
    print(int(duration / np.timedelta64(1, "h")), "hours")


def get_best(task, term, topn=1):
    files = glob.glob(f"results/{task}/*{term}*")
    data = []
    for file in files:
        with open(file, "r") as hundle:
            data_ = json.load(hundle)
            if "test" in data_:
                data.append((file, data_))
    if task == "dfd":
        data.sort(key=lambda x: x[1]["test"]["tauc"])
    elif task == "tfl":
        data.sort(key=lambda x: sum([x[1]["test"][y] for y in x[1]["test"] if y != "tloss"]))
    best = data[-topn]
    filename = best[0].split("/")[-1].split(".")[0]
    configuration = best[1]["config"]
    training = best[1]["training"]
    performance = best[1]["test"]
    return filename, configuration, training, performance


def get_ablation_plot(task, dataset, term, ylims, yticks_values):
    files = glob.glob(f"results/ablation/{task}_{dataset}*{term}*")
    data = []
    for file in files:
        with open(file, "r") as hundle:
            data_ = json.load(hundle)
            if "test" in data_:
                data.append(
                    {
                        term: data_["config"][term],
                        "performance": {x: data_["test"][x] for x in data_["test"] if "loss" not in x},
                    }
                )
    file = [x for x in glob.glob(f"ckpt/{task}/{task}_{dataset}*.json") if "whole" not in x][0]
    with open(file, "r") as hundle:
        best = json.load(hundle)
    data.append(
        {term: best["config"][term], "performance": {x: best["test"][x] for x in best["test"] if "loss" not in x}}
    )
    data.sort(key=lambda x: x[term] if term != "feature_pyramid" else str(x[term]))
    print(data)

    plt.figure(figsize=(7, 3))
    ax = plt.subplot(111)
    xaxis_values = [datum[term] for datum in data]
    metrics = [x for x in data[0]["performance"]]
    yaxis_values_all_metrics = []
    fontsize = 15
    markers = list(Line2D.markers.keys())
    for i, m in enumerate(metrics):
        yaxis_values = [datum["performance"][m] for datum in data]
        if term != "feature_pyramid":
            ax.plot(
                xaxis_values,
                yaxis_values,
                f"-{markers[i+2]}",
                label=m[1:].upper(),
                markerfacecolor="none",
                markeredgewidth=3,
                markersize=12,
                linewidth=2,
            )
        else:
            yaxis_values_all_metrics.append(yaxis_values)

    if term != "feature_pyramid":
        plt.xticks(xaxis_values, xaxis_values, fontsize=fontsize)
        plt.yticks(yticks_values, fontsize=fontsize, minor=False)
        plt.yticks((yticks_values[:-1] + yticks_values[1:]) / 2, minor=True)
        plt.ylim([ylims[0], ylims[1]])
        plt.grid(which="both")
        plt.subplots_adjust(left=0.08, right=0.73, top=0.95, bottom=0.12)
        plt.legend(bbox_to_anchor=(1, 1.05), fontsize=fontsize)
    else:
        width = 0.3
        colors = ["red", "blue"]
        for i in range(2):
            ax.bar(
                np.arange(len(metrics)) + (i - 1 / 2) * width,
                [y[i] for y in yaxis_values_all_metrics],
                width=width,
                color=colors[i],
            )
        plt.xticks(
            np.arange(len(metrics)),
            [m[1:].upper() for m in metrics],
            fontsize=fontsize,
            rotation=30 if task == "tfl" else 0,
        )
        plt.yticks(yticks_values, fontsize=fontsize)
        plt.ylim([ylims[0], ylims[1]])
        plt.grid(which="both")
        plt.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.25 if task == "tfl" else 0.12)
        legend_elements = ["FP use" if x else "No FP use" for x in xaxis_values]
        plt.legend(legend_elements, bbox_to_anchor=(0.15, 1.0), ncols=2, fontsize=fontsize)
    plt.savefig(f"results/figs/{task}_{dataset}_{term}.png")
    plt.close("all")
    return data


def get_scheduler_ablation_plot(task, dataset, ylims):
    fontsize = 14
    plt.figure(figsize=(7, 3))
    ax = plt.subplot(111)
    print(task, dataset)
    schedulers = {
        "none": "None",
        "reduceonplateau": "ReduceLROnPlateau",
        "step": "Step",
        "cosineanealing": "CosineAnnealing",
    }
    scores = []
    for scheduler in schedulers:
        _, _, _, performance = get_best(task=task, term=f"{dataset}_{scheduler}")
        if task == "dfd":
            score = performance["tauc"]
            scores.append(score)
            print(f"{scheduler}: {performance['tauc']:1.2f}/{performance['tacc']:1.2f}")
        else:
            score = sum([performance[y] for y in performance if y != "tloss"]) / (len(performance) - 1)
            scores.append(score)
            print(f"{scheduler}: {score:1.2f}")
    print()
    ax.bar(np.arange(len(schedulers)), scores)
    plt.xticks(np.arange(len(schedulers)), [schedulers[x] for x in schedulers], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim([ylims[0], ylims[1]])
    if task == "dfd":
        label = "AUC"
    else:
        label = "mean score (AP@p, AR@n)"
    plt.ylabel(label, fontsize=fontsize)
    plt.grid()
    plt.savefig(f"results/figs/schedulers_{task}_{dataset}.png")
    plt.close("all")


def plot_hyperparams_performance(metric, task, dataset):
    metric = f"t{metric}"
    files = glob.glob(f"results/{task}/*{dataset}*")
    performance = {}
    for file in files:
        with open(file, "r") as hundle:
            data = json.load(hundle)
            if "test" in data:
                layer_size = (
                    data["config"]["d_model"],
                    data["config"]["nhead"],
                    data["config"]["d_hid"],
                )
                num_layers = data["config"]["nlayers"]
                if layer_size not in performance:
                    performance[layer_size] = {}
                if num_layers not in performance[layer_size]:
                    performance[layer_size][num_layers] = [data["test"][metric]]
                else:
                    performance[layer_size][num_layers].append(data["test"][metric])

    plt.figure(figsize=(8, 3))
    fontsize = 14
    data = []
    xt = []
    layer_sizes = [x for x in performance]
    layer_sizes.sort(key=lambda x: math.prod(x))
    for layer_size in layer_sizes:
        num_layers_ = [x for x in performance[layer_size]]
        num_layers_.sort()
        for num_layers in num_layers_:
            xt.append(f"{num_layers}\n{layer_size}" if num_layers == 3 else f"{num_layers}")
            data.append(performance[layer_size][num_layers])
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    plt.boxplot(data, positions=positions)
    plt.xticks(positions, xt, fontsize=fontsize)
    yt = np.arange(85, 101, 5) if task == "tfl" else np.arange(0, 101, 20)
    plt.yticks(yt, fontsize=fontsize)
    plt.ylabel(metric[1:].upper(), fontsize=fontsize)
    plt.grid()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(f"results/figs/hp_{task}_{dataset}.png")


def performance_gathering(task, datasets, test_datasets, metrics):
    performance = []
    for dataset in datasets:
        folder = "results/generalization"
        filename = f"{task}_{dataset}"
        extention = "json"
        with open(f"{folder}/{filename}.{extention}", "r") as hundle:
            p = json.load(hundle)
            performance.append(
                ["/".join([f"{p[dataset][tdataset][m]:1.2f}" for m in metrics]) for tdataset in test_datasets]
            )
    return pd.DataFrame(
        data=performance,
        index=datasets,
        columns=test_datasets,
    )


def display_results():
    print("\nDFD (ACC/AP/AUC)")
    performance = performance_gathering(
        task="dfd",
        datasets=["fakeavceleb", "lavdf", "avdeepfake1m"],
        test_datasets=["fakeavceleb", "lavdf", "avdeepfake1m", "dfdc", "kodf"],
        metrics=["tacc", "tap", "tauc"],
    )
    print(performance)

    print("\nTFL (AP@0.5/AP@0.75/AP@0.9/AP@0.95/AR@100/AR@50/AR@30/AR@20/AR@10/AR@5)")
    performance = performance_gathering(
        task="tfl",
        datasets=["lavdf", "avdeepfake1m"],
        test_datasets=["lavdf", "avdeepfake1m"],
        metrics=[
            "tap@0.5",
            "tap@0.75",
            "tap@0.9",
            "tap@0.95",
            "tar@100",
            "tar@50",
            "tar@30",
            "tar@20",
            "tar@10",
            "tar@5",
        ],
    )
    print(performance)


def create_results_folder(name):
    folder_path = f"results/{name}"
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder '{folder_path}' created (or already existed).")


def get_interpretability_features(path, device, max_length):
    data = np.load(path, allow_pickle=True)
    video_features = torch.tensor(data["video_features"])
    audio_features = torch.tensor(data["audio_features"])
    t = min(video_features.shape[0], audio_features.shape[0], max_length)
    video_features = (
        torch.concat(
            [
                video_features[:t, :],
                torch.zeros([max_length - t, video_features.shape[1]]),
            ]
        )
        .unsqueeze(0)
        .to(device)
    )
    audio_features = (
        torch.concat(
            [
                audio_features[:t, :],
                torch.zeros([max_length - t, audio_features.shape[1]]),
            ]
        )
        .unsqueeze(0)
        .to(device)
    )
    return video_features, audio_features, t


def create_interpretability_figures():
    device = "cpu"
    max_length = 600
    workers = 1
    batch_size = 64
    num_videos = 30
    models = [
        {
            "ckpt": "ckpt/tfl/tfl_lavdf_reduceonplateau_256_8_2_5_True.pth",
            "json": "ckpt/tfl/tfl_lavdf_reduceonplateau_256_8_2_5_True.json",
            "task": "tfl",
            "dataset": "lavdf",
        },
    ]

    with open("data/LAV-DF_emb/metadata.min.json", "r") as f:
        metadata = json.load(f)[:num_videos]

    for video_metadata in metadata:
        print(video_metadata)
        video_name = video_metadata["file"][5:].replace(".mp4", "")
        video_fake_parts = video_metadata["fake_periods"]
        print(video_name, video_fake_parts)
        video_path = "data/LAV-DF_emb/" + video_metadata["file"].replace(".mp4", "/mediapipe/features.npz")
        video, audio, length = get_interpretability_features(video_path, device, max_length)
        for m in models:
            ckpt_path = m["ckpt"]
            with open(m["json"], "r") as hundle:
                data = json.load(hundle)
            configuration = data["config"]
            experiment = Experiment(
                dataset="none",
                task=m["task"],
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
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location=torch.device(device))
            model.load_state_dict(checkpoint["model"])
            model.eval()
            p, z = model([video, audio])
            sim = torch.nn.functional.cosine_similarity(
                z.squeeze()[:, :max_length, :], z.squeeze()[:, max_length:, :], dim=-1
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            kernel_size = 15
            kernel = np.ones(kernel_size) / kernel_size
            mean_p = (
                torch.sigmoid(torch.stack((p.squeeze()[:, :max_length, 0], p.squeeze()[:, max_length:, 0]), dim=-1))
                .max(dim=-1)[0][:, :length]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            std_p = (
                torch.sigmoid(torch.stack((p.squeeze()[:, :max_length, 0], p.squeeze()[:, max_length:, 0]), dim=-1))
                .max(dim=-1)[0][:, :length]
                .std(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            mean_cs = sim[:, :length].mean(dim=0).detach().cpu().numpy()
            std_cs = sim[:, :length].std(dim=0).detach().cpu().numpy()
            y_min = min(np.nanmin(mean_cs - std_cs), 0.0)
            y_max = np.nanmax(mean_cs + std_cs)
            ax.plot(mean_cs, linewidth=2, label="cosine sim.")
            ax.fill_between(np.arange(mean_cs.shape[0]), mean_cs - std_cs, mean_cs + std_cs, alpha=0.4)
            ax.plot(mean_p, linewidth=2, label="$\\max_{m}(\\hat{a}^m_{\\phi})$", color="indigo", linestyle="--")
            ax.fill_between(np.arange(mean_p.shape[0]), mean_p - std_p, mean_p + std_p, alpha=0.8, color="thistle")
            if video_fake_parts:
                for i, fake_part in enumerate(video_fake_parts):
                    ax.add_patch(
                        plt.Rectangle(
                            (fake_part[0] * 25, y_min),
                            (fake_part[1] - fake_part[0]) * 25,
                            y_max - y_min,
                            ec="none",
                            fc="r",
                            alpha=0.4,
                            label="fake" if i == 0 else None,
                        )
                    )
            fontsize = 14
            current_ticks = np.linspace(0, length, 7)
            plt.xticks(current_ticks, [np.round(x / 25, 2) for x in current_ticks], fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.ylim([y_min, y_max])
            plt.grid()

            plt.xlabel("timestamp [sec]", fontsize=fontsize)
            fig.subplots_adjust(bottom=0.2)
            plt.legend(loc="lower right", fontsize=fontsize)
            plt.savefig(f"results/interpretability/{m['dataset']}_{video_name}.png")
            plt.close(fig)


def plot_robustness_performance(performance, competitive, modality, nrows, ncols):
    """
    Takes a performance dictionary and creates a figure with 8 subplots,
    depicting AP and AUC vs levels for each type and the average.

    Args:
        performance (dict): A dictionary containing performance metrics (AP, AUC)
                          for each type and level.
    """
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["lines.markersize"] = 10
    plt.rcParams["lines.markeredgewidth"] = 2
    plt.rcParams["lines.markerfacecolor"] = "none"
    type_list = list(performance.keys())
    level_list = list(list(performance.values())[0].keys())

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    extra_plot_data = None
    for i, type_name in enumerate(type_list):
        ap_values = []
        auc_values = []
        ap_avff = []
        auc_avff = []
        auc_realforensics = []
        for level in level_list:
            ap_values.append(
                performance[type_name][level]["AP"] * 100 if performance[type_name][level]["AP"] != -1 else float("nan")
            )
            auc_values.append(
                performance[type_name][level]["AUC"] * 100
                if performance[type_name][level]["AUC"] != -1
                else float("nan")
            )
            if modality == "visual":
                auc_avff.append(competitive[type_name]["AVFF"][level])
                auc_realforensics.append(competitive[type_name]["RealForensics"][level])
            else:
                ap_avff.append(competitive[type_name]["AP"][level])
                auc_avff.append(competitive[type_name]["AUC"][level])

        axes[i].plot(level_list, ap_values, marker="s", label="AP (DiMoDif)")
        axes[i].plot(level_list, auc_values, marker="^", label="AUC (DiMoDif)")
        if modality == "audio":
            axes[i].plot(level_list, ap_avff, marker="s", label="AP (AVFF)")
        axes[i].plot(level_list, auc_avff, marker="o", label="AUC (AVFF)")
        if modality == "visual":
            axes[i].plot(level_list, auc_realforensics, marker="v", label="AUC (RealForensics)")
        axes[i].set_title(f"{type_dict[type_name]}")
        axes[i].set_xlabel("Intensity")
        axes[i].set_ylim((73 if i != 3 else 48) if modality == "visual" else 73, 102)
        axes[i].grid()
        if type_name == "GB" and modality == "visual":
            extra_plot_data = (level_list, ap_values, auc_values, auc_avff, auc_realforensics)
        elif type_name == "GN" and modality == "audio":
            extra_plot_data = (level_list, ap_values, auc_values, ap_avff, auc_avff)

    # Calculate and plot average performance
    if modality == "visual":
        avg_ap = []
        avg_auc = []
        avg_auc_avff = []
        avg_auc_realforensics = []
        for level in level_list:
            ap_sum = 0
            auc_sum = 0
            auc_avff_sum = 0
            auc_realforensics_sum = 0
            count = 0
            for type_name in type_list:
                if performance[type_name][level]["AP"] != -1:
                    ap_sum += performance[type_name][level]["AP"]
                    auc_sum += performance[type_name][level]["AUC"]
                    auc_avff_sum += competitive[type_name]["AVFF"][level]
                    auc_realforensics_sum += competitive[type_name]["RealForensics"][level]
                    count += 1
            avg_ap.append((ap_sum / count) * 100 if count > 0 else float("nan"))
            avg_auc.append((auc_sum / count) * 100 if count > 0 else float("nan"))
            avg_auc_avff.append(auc_avff_sum / len(type_list))
            avg_auc_realforensics.append(auc_realforensics_sum / len(type_list))
        # print(f"Average AP across perturbations at intensity level 5: {avg_ap[-1]:1.1f}")
        axes[7].plot(level_list, avg_ap, marker="s", label="AP (DiMoDif)")
        axes[7].plot(level_list, avg_auc, marker="^", label="AUC (DiMoDif)")
        axes[7].plot(level_list, avg_auc_avff, marker="o", label="AUC (AVFF)")
        axes[7].plot(level_list, avg_auc_realforensics, marker="v", label="AUC (RealForensics)")
        axes[7].set_title("Average")
        axes[7].set_xlabel("Intensity")
        axes[7].set_ylim(73, 102)
        axes[7].grid()

    handles, labels = axes[0].get_legend_handles_labels()  # Get legend handles and labels from the first subplot
    fig.legend(
        handles,
        labels,
        loc="upper center",  # Position the legend between the rows
        bbox_to_anchor=(0.5, 0.5405),  # Adjust the position (x=0.5 centers it horizontally)
        ncol=4,  # Number of columns in the legend
        fontsize=18 if modality == "visual" else 13,  # Adjust font size
    )

    plt.tight_layout(h_pad=3.5)
    plt.savefig(f"results/robustness/{modality}_performance.png")
    if extra_plot_data and modality == "visual":
        level_list, ap_values, auc_values, auc_avff, auc_realforensics = extra_plot_data
        plt.figure(figsize=(5, 4))
        plt.plot(level_list, ap_values, marker="s", label="AP (DiMoDif)")
        plt.plot(level_list, auc_values, marker="^", label="AUC (DiMoDif)")
        plt.plot(level_list, auc_avff, marker="o", label="AUC (AVFF)")
        plt.plot(level_list, auc_realforensics, marker="v", label="AUC (RealForensics)")
        plt.title("Gaussian Blur")
        plt.xlabel("Intensity")
        plt.ylim(73, 102)
        plt.grid()
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"results/robustness/{modality}_performance_gb.png")
    elif extra_plot_data and modality == "audio":
        level_list, ap_values, auc_values, ap_avff, auc_avff = extra_plot_data
        plt.figure(figsize=(5, 4))
        plt.plot(level_list, ap_values, marker="s", label="AP (DiMoDif)")
        plt.plot(level_list, auc_values, marker="^", label="AUC (DiMoDif)")
        plt.plot(level_list, ap_avff, marker="o", label="AP (AVFF)")
        plt.plot(level_list, auc_avff, marker="v", label="AUC (AVFF)")
        plt.title("Gaussian Noise")
        plt.xlabel("Intensity")
        plt.ylim(73, 102)
        plt.grid()
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"results/robustness/{modality}_performance_gn.png")


def plot_robustness_common(visual_performance, audio_performance, visual_competitive, audio_competitive, levels):
    dimodif_visual_auc = [visual_performance["GB"][level]["AUC"] * 100 for level in levels]
    dimodif_audio_auc = [audio_performance["GN"][level]["AUC"] * 100 for level in levels]
    avff_visual_auc = [visual_competitive["GB"]["AVFF"][level] for level in levels]
    avff_audio_auc = [audio_competitive["GN"]["AUC"][level] for level in levels]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(levels, dimodif_visual_auc, marker="s", label="DiMoDif (visual: Gaussian blur)")
    ax.plot(levels, dimodif_audio_auc, marker="^", label="DiMoDif (audio: Gaussian noise)")
    ax.plot(levels, avff_visual_auc, marker="o", label="AVFF (visual: Gaussian blur)")
    ax.plot(levels, avff_audio_auc, marker="v", label="AVFF (audio: Gaussian noise)")
    plt.xlabel("Intensity")
    plt.ylabel("AUC")
    plt.ylim(68, 102)
    plt.grid()
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/robustness/common_performance.png")


def get_robustness_competitive_performance(modality="visual"):
    if modality == "visual":
        competitive = {
            "CS": {
                "RealForensics": {"0": 94.6, "1": 92.6, "2": 92.7, "3": 92.7, "4": 92.5, "5": 92.3},
                "AVFF": {"0": 99.1, "1": 98.8, "2": 98.6, "3": 98.7, "4": 98.9, "5": 98.4},
            },
            "CC": {
                "RealForensics": {"0": 94.6, "1": 92.8, "2": 92.5, "3": 92.5, "4": 92.0, "5": 90.5},
                "AVFF": {"0": 99.1, "1": 98.5, "2": 98.0, "3": 95.5, "4": 93.5, "5": 86.0},
            },
            "BW": {
                "RealForensics": {"0": 94.6, "1": 91.5, "2": 88.7, "3": 84.7, "4": 82.0, "5": 78.0},
                "AVFF": {"0": 99.1, "1": 98.5, "2": 97.5, "3": 95.5, "4": 94.0, "5": 93.0},
            },
            "GNC": {
                "RealForensics": {"0": 94.6, "1": 83.5, "2": 79.5, "3": 71.0, "4": 65.5, "5": 59.3},
                "AVFF": {"0": 99.1, "1": 98.2, "2": 98.6, "3": 96.4, "4": 90.0, "5": 71.0},
            },
            "GB": {
                "RealForensics": {"0": 94.6, "1": 89.3, "2": 88.2, "3": 85.6, "4": 83.6, "5": 82.0},
                "AVFF": {"0": 99.1, "1": 95.8, "2": 93.1, "3": 88.1, "4": 84.5, "5": 81.1},
            },
            "JPEG": {
                "RealForensics": {"0": 94.6, "1": 90.5, "2": 89.8, "3": 89.1, "4": 88.7, "5": 87.0},
                "AVFF": {"0": 99.1, "1": 96.3, "2": 93.9, "3": 92.6, "4": 90.5, "5": 87.0},
            },
            "VC": {
                "RealForensics": {"0": 94.6, "1": 90.5, "2": 89.5, "3": 87.0, "4": 84.5, "5": 82.0},
                "AVFF": {"0": 99.1, "1": 98.8, "2": 98.7, "3": 98.0, "4": 97.0, "5": 96.0},
            },
        }
    else:
        competitive = {
            "GN": {
                "AUC": {"0": 99.1, "1": 99.0, "2": 98.0, "3": 97.0, "4": 94.1, "5": 93},
                "AP": {"0": 96.8, "1": 96.7, "2": 96.0, "3": 94.0, "4": 93.2, "5": 90.4},
            },
            "PS": {
                "AUC": {"0": 99.1, "1": 98.2, "2": 97.3, "3": 97.2, "4": 97.1, "5": 97.0},
                "AP": {"0": 97.0, "1": 94.1, "2": 90.3, "3": 90.7, "4": 89.5, "5": 89},
            },
            "RV": {
                "AUC": {"0": 99.1, "1": 98.9, "2": 98.7, "3": 98.5, "4": 98.3, "5": 98.3},
                "AP": {"0": 97.0, "1": 96.5, "2": 96.4, "3": 95.9, "4": 95.5, "5": 94.8},
            },
            "AC": {
                "AUC": {"0": 99.1, "1": 99.0, "2": 98.8, "3": 97.7, "4": 95.3, "5": 94.5},
                "AP": {"0": 97.0, "1": 96.5, "2": 95.5, "3": 87.0, "4": 84.0, "5": 82.5},
            },
        }
    return competitive


def get_robustness_performance(type_list, modality):
    level_list = ["1", "2", "3", "4", "5"]
    performance = {}
    for type in type_list:
        performance[type] = {"0": {"AP": 0.9999258282869707, "AUC": 0.9971045169733503}}
        for level in level_list:
            print(f"Type {type} Level {level}")
            filename = f"utils/robustness/{modality}_{type}_{level}.json"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    results = json.load(f)
                y_true = [result["label"] for result in results]
                y_pred = [result["prediction"] for result in results]
                if 0 in y_true and 1 in y_true:
                    ap = average_precision_score(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_pred)
                    print(f"Number of videos: {len(results)} AP: {ap*100:.2f} AUC: {auc*100:.2f}")
                    performance[type][level] = {"AP": ap, "AUC": auc}
                else:
                    print(f"Not enough labels for {type} {level}. Number of videos: {len(results)}")
                    performance[type][level] = {"AP": -1, "AUC": -1}
            else:
                print(f"File not found: {filename}")
                performance[type][level] = {"AP": -1, "AUC": -1}
                continue

    aps = []
    for type in type_list:
        aps.append(performance[type]["5"]["AP"] * 100)
    print(f"Avgerage AP across perturbations at intensity level 5: {sum(aps)/len(aps):1.2f}")

    aucs = []
    for type in type_list:
        aucs.append(performance[type]["5"]["AUC"] * 100)
    print(f"Avgerage AUC across perturbations at intensity level 5: {sum(aucs)/len(aucs):1.2f}")
    return performance


if __name__ == "__main__":
    create_results_folder("figs")

    print("\nMain evaluation results:")
    display_results()

    print("\nWindow size ablation")
    get_ablation_plot("dfd", "fakeavceleb", "win_size", [75, 102], np.arange(75, 101, 5))
    get_ablation_plot("dfd", "lavdf", "win_size", [75, 102], np.arange(75, 101, 5))
    get_ablation_plot("dfd", "avdeepfake1m", "win_size", [75, 102], np.arange(75, 101, 5))
    get_ablation_plot("tfl", "lavdf", "win_size", [0, 105], np.arange(0, 101, 20))
    get_ablation_plot("tfl", "avdeepfake1m", "win_size", [0, 105], np.arange(0, 101, 20))

    print("\nalpha ablation")
    get_ablation_plot("tfl", "lavdf", "alpha", [0, 105], np.arange(0, 101, 20))
    get_ablation_plot("tfl", "avdeepfake1m", "alpha", [0, 105], np.arange(0, 101, 20))

    print("\nfeature pyramid ablation")
    get_ablation_plot("dfd", "fakeavceleb", "feature_pyramid", [0, 105], np.arange(0, 101, 20))
    get_ablation_plot("dfd", "lavdf", "feature_pyramid", [0, 105], np.arange(0, 101, 20))
    get_ablation_plot("dfd", "avdeepfake1m", "feature_pyramid", [0, 105], np.arange(0, 101, 20))
    get_ablation_plot("tfl", "lavdf", "feature_pyramid", [0, 105], np.arange(0, 101, 20))
    get_ablation_plot("tfl", "avdeepfake1m", "feature_pyramid", [0, 105], np.arange(0, 101, 20))

    print("\nScheduler ablation")
    get_scheduler_ablation_plot(task="dfd", dataset="fakeavceleb", ylims=[90, 100])
    get_scheduler_ablation_plot(task="dfd", dataset="lavdf", ylims=[90, 100])
    get_scheduler_ablation_plot(task="dfd", dataset="avdeepfake1m", ylims=[90, 100])
    get_scheduler_ablation_plot(task="tfl", dataset="lavdf", ylims=[60, 85])
    get_scheduler_ablation_plot(task="tfl", dataset="avdeepfake1m", ylims=[60, 85])

    for dataset in ["fakeavceleb", "lavdf", "avdeepfake1m"]:
        plot_hyperparams_performance(metric="acc", task="dfd", dataset=dataset)
    for dataset in ["lavdf", "avdeepfake1m"]:
        plot_hyperparams_performance(metric="ap@0.5", task="tfl", dataset=dataset)

    print("\nInterpretability figures")
    create_results_folder("interpretability")
    create_interpretability_figures()

    create_results_folder("robustness")
    print("\nVisual robustness")
    type_list = ["CS", "CC", "BW", "GNC", "GB", "JPEG", "VC"]
    type_dict = {
        "CS": "Saturation",
        "CC": "Contrast",
        "BW": "Block-wise",
        "GNC": "Gaussian Noise",
        "GB": "Gaussian Blur",
        "JPEG": "JPEG Compression",
        "VC": "Video Compression",
    }
    visual_performance = get_robustness_performance(type_list, "visual")
    visual_competitive = get_robustness_competitive_performance(modality="visual")
    plot_robustness_performance(visual_performance, visual_competitive, modality="visual", nrows=2, ncols=4)

    print("\nAudio robustness")
    type_list = ["GN", "PS", "RV", "AC"]
    type_dict = {
        "GN": "Gaussian Noise",
        "PS": "Pitch Shift",
        "RV": "Reverberence",
        "AC": "Audio Compression",
    }
    audio_performance = get_robustness_performance(type_list, "audio")
    audio_competitive = get_robustness_competitive_performance(modality="audio")
    plot_robustness_performance(audio_performance, audio_competitive, modality="audio", nrows=2, ncols=2)

    plot_robustness_common(
        visual_performance,
        audio_performance,
        visual_competitive,
        audio_competitive,
        levels=["0", "1", "2", "3", "4", "5"],
    )

    print("\nGPU hours:")
    get_gpu_hours()
