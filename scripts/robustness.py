import argparse
import os
import random
import cv2
import math
import numpy as np
import torch
import json
import subprocess
import ffmpeg

import torchaudio
from pydub import AudioSegment
from pysndfx import AudioEffectsChain

from src.datasets import FakeAVCeleb
from src.robustness import AVSRDataLoader, LandmarksDetector, get_pipeline_obj, extract_data
from src.models import AVDeepFakeDetector

MAX_FRAMES_PER_SEGMENT = 600


def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

    return img_ycbcr


def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    # to [0, 1]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    # to [0, 1]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_bgr


def color_saturation(img, param):
    ycbcr = bgr2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2bgr(ycbcr).astype(np.uint8)

    return img


def color_contrast(img, param):
    img = img.astype(np.float32) * param
    img = img.astype(np.uint8)

    return img


def block_wise(img, param):
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    for i in range(param):
        r_w = random.randint(0, img.shape[1] - 1 - width)
        r_h = random.randint(0, img.shape[0] - 1 - width)
        img[r_h : r_h + width, r_w : r_w + width, :] = block

    return img


def gaussian_noise_color(img, param):
    ycbcr = bgr2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) * np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2bgr(b)
    img = np.clip(b, 0, 255).astype(np.uint8)

    return img


def gaussian_blur(img, param):
    img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)

    return img


def jpeg_compression(img, param):
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))

    return img


def video_compression(vid_in, vid_out, param):
    cmd = f"/usr/bin/ffmpeg -i '{vid_in}' -crf {param} -y {vid_out}"
    os.system(cmd)


def get_distortion_parameter(type, level):
    param_dict = dict()  # a dict of list
    param_dict["CS"] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse
    param_dict["CC"] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse
    param_dict["BW"] = [16, 32, 48, 64, 80]  # larger, worse
    param_dict["GNC"] = [0.001, 0.002, 0.005, 0.01, 0.05]  # larger, worse
    param_dict["GB"] = [7, 9, 13, 17, 21]  # larger, worse
    param_dict["JPEG"] = [2, 3, 4, 5, 6]  # larger, worse
    param_dict["VC"] = [30, 32, 35, 38, 40]  # larger, worse

    # level starts from 1, list starts from 0
    return param_dict[type][level - 1]


def get_distortion_function(type):
    func_dict = dict()  # a dict of function
    func_dict["CS"] = color_saturation
    func_dict["CC"] = color_contrast
    func_dict["BW"] = block_wise
    func_dict["GNC"] = gaussian_noise_color
    func_dict["GB"] = gaussian_blur
    func_dict["JPEG"] = jpeg_compression
    func_dict["VC"] = video_compression

    return func_dict[type]


def get_audio_distortion_parameter(type, level):
    param_dict = dict()  # a dict of list
    param_dict["GN"] = [40, 30, 20, 15, 10]
    param_dict["PS"] = [2, 4, 6, 8, 10]
    param_dict["RV"] = [20, 40, 60, 80, 100]
    param_dict["AC"] = ["320k", "256k", "192k", "128k", "64k"]

    # level starts from 1, list starts from 0
    return param_dict[type][level - 1]


def get_audio_distortion_function(type):
    func_dict = dict()  # a dict of function
    func_dict["GN"] = apply_gaussian_noise
    func_dict["PS"] = apply_pitch_shift
    func_dict["RV"] = apply_reverberance
    func_dict["AC"] = apply_audio_compression

    return func_dict[type]


def get_audio_info(video_file):
    """Gets audio information (bitrate, channels, sample rate) from a video file."""
    try:
        probe = ffmpeg.probe(video_file)
        audio_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None)
        if audio_stream:
            bit_rate = audio_stream.get("bit_rate", "320k")  # default to 320k if missing
            channels = audio_stream.get("channels", 2)  # default to stereo if missing
            sample_rate = audio_stream.get("sample_rate", 44100)  # default to 44.1kHz if missing
            return bit_rate, channels, sample_rate
        else:
            return "320k", 2, 44100  # Default values if no audio stream found
    except ffmpeg.Error as e:
        print(f"Error probing file: {e.stderr.decode()}")
        return "320k", 2, 44100  # Default to prevent further errors.


def extract_audio(video_file, audio_file):
    """Extracts audio from a video file using ffmpeg with original audio properties."""
    bit_rate, channels, sample_rate = get_audio_info(video_file)
    command = f"/usr/bin/ffmpeg -i '{video_file}' -ab {bit_rate} -ac {channels} -ar {sample_rate} -vn {audio_file}"
    subprocess.call(command, shell=True)
    return audio_file


def apply_gaussian_noise(audio_file, snr):
    """Applies Gaussian noise based on intensity level."""
    waveform, sample_rate = torchaudio.load(audio_file)
    noise = torch.randn_like(waveform)
    signal_power = torch.sum(waveform**2)
    noise_power = torch.sum(noise**2)
    alpha = torch.sqrt(signal_power / (snr * noise_power))
    noisy_signal = waveform + alpha * noise
    torchaudio.save(audio_file, noisy_signal, sample_rate)


def apply_pitch_shift(audio_file, n_steps):
    """Applies pitch shift based on intensity level."""
    if random.random() < 0.5:
        n_steps = -n_steps
    waveform, sample_rate = torchaudio.load(audio_file)
    effects = torchaudio.transforms.PitchShift(sample_rate, n_steps)
    perturbed_waveform = effects(waveform)
    torchaudio.save(audio_file, perturbed_waveform, sample_rate)


def apply_reverberance(audio_file, reverb):
    """Applies reverberance based on intensity level."""
    fx = AudioEffectsChain().reverb(reverberance=reverb)
    output_file = audio_file.replace(".wav", "_reverb.wav")
    fx(audio_file, output_file)


def apply_audio_compression(audio_file, bitrate):
    """Applies audio compression based on intensity level."""
    audio = AudioSegment.from_wav(audio_file)
    audio.export(audio_file, format="mp3", bitrate=bitrate)


def distortion_aud(vid_in, aud_out, type="random", level="random"):
    # get distortion type
    if type == "random":
        dist_types = ["GN", "PS", "RV", "AC"]
        type_id = random.randint(0, 3)
        dist_type = dist_types[type_id]
    else:
        dist_type = type

    # get distortion level
    if level == "random":
        dist_level = random.randint(1, 5)
    else:
        dist_level = int(level)

    # extract audio
    extract_audio(vid_in, aud_out)

    # get distortion parameter
    dist_param = get_audio_distortion_parameter(dist_type, dist_level)
    dist_function = get_audio_distortion_function(dist_type)
    dist_function(aud_out, dist_param)


def distortion_vid(vid_in, vid_out, type="random", level="random"):
    # get distortion type
    if type == "random":
        dist_types = ["CS", "CC", "BW", "GNC", "GB", "JPEG", "VC"]
        type_id = random.randint(0, 6)
        dist_type = dist_types[type_id]
    else:
        dist_type = type

    # get distortion level
    if level == "random":
        dist_level = random.randint(1, 5)
    else:
        dist_level = int(level)

    # get distortion parameter
    dist_param = get_distortion_parameter(dist_type, dist_level)

    # get distortion function
    dist_function = get_distortion_function(dist_type)

    # apply distortion
    if dist_type == "VC":
        dist_function(vid_in, vid_out, dist_param)
    else:
        # extract frames
        vid = cv2.VideoCapture(vid_in)
        fps = vid.get(cv2.CAP_PROP_FPS)
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_out, fourcc, fps, (w, h))
        while True:
            success, frame = vid.read()
            if not success:
                break
            writer.write(dist_function(frame, dist_param))
        vid.release()
        writer.release()


def get_model(device):
    json_path = "ckpt/dfd/dfd_fakeavceleb_reduceonplateau_64_4_1_3_True_none.json"
    ckpt_path = "ckpt/dfd/dfd_fakeavceleb_reduceonplateau_64_4_1_3_True_none.pth"
    with open(json_path, "r") as hundle:
        configuration = json.load(hundle)["config"]
    model = AVDeepFakeDetector(
        task=configuration["task"],
        max_length=configuration["max_length"],
        d_model=configuration["d_model"],
        nhead=configuration["nhead"],
        d_hid=configuration["d_hid"],
        nlayers=configuration["nlayers"],
        win_size=configuration["win_size"],
        feature_pyramid=configuration["feature_pyramid"],
        device=device,
    )
    model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def transform_features(features, length, device):
    return torch.cat(
        [features[:length, :], torch.zeros([MAX_FRAMES_PER_SEGMENT - length, features.shape[1]]).to(device)]
    ).unsqueeze(0)


def adjust_features(video_features, audio_features, device):
    t = min(video_features.shape[0], audio_features.shape[0], MAX_FRAMES_PER_SEGMENT)
    return transform_features(video_features, t, device), transform_features(audio_features, t, device)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_features(audio, video, pipeline_audio, pipeline_video):
    return extract_data(pipeline_audio, pipeline_video, audio, video)


def get_prediction(
    original,
    perturbed,
    detector,
    videoloader,
    audioloader,
    model,
    pipeline_audio,
    pipeline_video,
    visual_dist=True,
    device="cuda",
):
    landmarks = detector(perturbed if visual_dist else original)
    landmarks = [None if x is None else x[0] for x in landmarks]
    if landmarks:
        video, _ = videoloader.load_data(perturbed if visual_dist else original, landmarks=landmarks)
        audio, _ = audioloader.load_data(original if visual_dist else perturbed, landmarks=landmarks)
        data = get_features(audio, video, pipeline_audio, pipeline_video)
        video_features, audio_features = adjust_features(data["video_features"], data["audio_features"], device)
        with torch.no_grad():
            outputs = model([video_features, audio_features])
        logit = np.squeeze(np.array(outputs[0].cpu()).mean(axis=1).max(axis=-1)).item()
        return sigmoid(logit)
    else:
        return None


def visual(args):
    device = "cuda"
    type_list = ["CS", "CC", "BW", "GNC", "GB", "JPEG", "VC"]
    level_list = ["1", "2", "3", "4", "5"]
    type_level = [(x, y) for x in type_list for y in level_list]
    type, level = type_level[int(args.index)]
    detector = LandmarksDetector()
    videoloader = AVSRDataLoader(modality="video")
    audioloader = AVSRDataLoader(modality="audio")
    pipeline_audio = get_pipeline_obj("audio", device)
    pipeline_video = get_pipeline_obj("video", device)
    model = get_model(device)
    video_paths = FakeAVCeleb(split="test", max_length=600, without="none", showsize=True).videos
    random.seed(0)
    random.shuffle(video_paths)
    filename = f"utils/robustness/visual_{type}_{level}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results = json.load(f)
        processed_vids = [result["filename"] for result in results]
    else:
        results = []
        processed_vids = []
    for video_path in video_paths:
        vid_out = f"tmp/{np.random.randint(0, 1000000):07d}.mp4"
        vid_in = (
            video_path[0]
            .replace("data/FakeAVCeleb_emb", "data/FakeAVCeleb_v1.2")
            .replace("data/VoxCeleb2_emb", "data/VoxCeleb2")
            .replace("/mediapipe/features.npz", ".mp4")
        )
        print(vid_in, vid_out)
        if vid_in not in processed_vids:
            label = int(video_path[1] == 1 or video_path[2] == 1)
            distortion_vid(vid_in, vid_out, type, level)
            prediction = get_prediction(
                original=vid_in,
                perturbed=vid_out,
                detector=detector,
                videoloader=videoloader,
                audioloader=audioloader,
                model=model,
                pipeline_audio=pipeline_audio,
                pipeline_video=pipeline_video,
                visual_dist=True,
                device=device,
            )
            results.append({"filename": vid_in, "label": label, "prediction": prediction})
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            os.remove(vid_out)


def audio(args):
    device = "cuda"
    type_list = ["GN", "PS", "RV", "AC"]
    level_list = ["1", "2", "3", "4", "5"]
    type_level = [(x, y) for x in type_list for y in level_list]
    type, level = type_level[int(args.index)]
    print(type, level)
    detector = LandmarksDetector()
    videoloader = AVSRDataLoader(modality="video")
    audioloader = AVSRDataLoader(modality="audio")
    pipeline_audio = get_pipeline_obj("audio", device)
    pipeline_video = get_pipeline_obj("video", device)
    model = get_model(device)
    video_paths = FakeAVCeleb(split="test", max_length=600, without="none", showsize=True).videos
    random.seed(0)
    random.shuffle(video_paths)
    filename = f"utils/robustness/audio_{type}_{level}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results = json.load(f)
        processed_vids = [result["filename"] for result in results]
    else:
        results = []
        processed_vids = []
    for video_path in video_paths:
        aud_out = f"tmp/{np.random.randint(0, 1000000):07d}.wav"
        vid_in = (
            video_path[0]
            .replace("data/FakeAVCeleb_emb", "data/FakeAVCeleb_v1.2")
            .replace("data/VoxCeleb2_emb", "data/VoxCeleb2")
            .replace("/mediapipe/features.npz", ".mp4")
        )
        print(vid_in, aud_out)
        if vid_in not in processed_vids:
            label = int(video_path[1] == 1 or video_path[2] == 1)
            distortion_aud(vid_in, aud_out, type, level)
            prediction = get_prediction(
                original=vid_in,
                perturbed=aud_out,
                detector=detector,
                videoloader=videoloader,
                audioloader=audioloader,
                model=model,
                pipeline_audio=pipeline_audio,
                pipeline_video=pipeline_video,
                visual_dist=False,
                device=device,
            )
            if prediction is not None:
                results.append({"filename": vid_in, "label": label, "prediction": prediction})
                with open(filename, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                print("No landmarks detected, skipping this video.", vid_in)
            os.remove(aud_out)
            if type == "RV":
                os.remove(aud_out.replace(".wav", "_reverb.wav"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a distortion to videos and get DiMoDif predictions.")
    parser.add_argument(
        "-m",
        "--modality",
        help="modality to apply perturbations",
        default="visual",
    )
    parser.add_argument(
        "-i",
        "--index",
        help="slurm array task id",
        default=0,
    )
    args = parser.parse_args()
    os.makedirs("tmp", exist_ok=True)
    if args.modality == "visual":
        visual(args)
    elif args.modality == "audio":
        audio(args)
    else:
        raise ValueError("modality should be visual or audio")
