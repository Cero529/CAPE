import os
import pandas as pd
from pipelines.data.data_module import AVSRDataLoader
from speechbrain.inference.classifiers import EncoderClassifier
import json

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
root_dir = "datasets/VoxCeleb2/dev/mp4"
metadata = pd.read_csv(
    "datasets/VoxCeleb2/vox2_meta.csv",
    sep=" ,",
    engine="python",
)
dataloader = AVSRDataLoader(modality="audio")
video_paths_m = []
video_paths_f = []
for root, dirs, files in os.walk(root_dir, topdown=False):
    identity = os.path.basename(os.path.dirname(root))
    if identity != "mp4":
        gender, split_tmp = metadata[metadata["VoxCeleb2 ID"] == identity].values.tolist()[0][-2:]
        for name in files:
            waveform, sample_rate = dataloader.load_audio(os.path.join(root, name))
            prediction = language_id.classify_batch(waveform)
            print(
                f"[m:{len(video_paths_m)}/10000,f:{len(video_paths_f)}/10000] [{os.path.join(root, name)}-->{prediction[3]}]"
            )
            if "en: English" in prediction[3]:
                if gender == "m" and split_tmp == "dev" and len(video_paths_m) < 10000:
                    video_paths_m.append(os.path.join(root, name))
                if gender == "f" and split_tmp == "dev" and len(video_paths_f) < 10000:
                    video_paths_f.append(os.path.join(root, name))
        with open("voxceleb2_eng.json", "w") as hundle:
            json.dump(video_paths_m + video_paths_f, hundle, indent=2)
        if len(video_paths_m) == 10000 and len(video_paths_f) == 10000:
            break
