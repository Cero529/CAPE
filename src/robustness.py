import torch
import torchaudio
import torchvision
import numpy as np
import mediapipe as mp
import cv2
import json
import argparse

from src.models import AVDeepFakeDetector

from distutils.util import strtobool
import logging
import torch

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E(torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="how to initialize transformer parameters",
        )
        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv3d", "conv2d", "conv1d", "linear", "embed"],
            help="transformer input layer type",
        )
        group.add_argument(
            "--transformer-encoder-attn-layer-type",
            type=str,
            default="mha",
            choices=["mha", "rel_mha", "legacy_rel_mha"],
            help="transformer encoder attention layer type",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate",
            default=None,
            type=float,
            help="dropout in transformer attention. use --dropout-rate if None is set",
        )
        group.add_argument(
            "--transformer-lr",
            default=10.0,
            type=float,
            help="Initial value of learning rate",
        )
        group.add_argument(
            "--transformer-warmup-steps",
            default=25000,
            type=int,
            help="optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-length-normalized-loss",
            default=True,
            type=strtobool,
            help="normalize loss by length",
        )
        group.add_argument(
            "--dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate for the encoder",
        )
        group.add_argument(
            "--macaron-style",
            default=False,
            type=strtobool,
            help="Whether to use macaron style for positionwise layer",
        )
        # -- input
        group.add_argument(
            "--a-upsample-ratio",
            default=1,
            type=int,
            help="Upsample rate for audio",
        )
        group.add_argument(
            "--relu-type",
            default="swish",
            type=str,
            help="the type of activation layer",
        )
        # Encoder
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers (for shared recognition part " "in multi-speaker asr mode)",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=300,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--use-cnn-module",
            default=False,
            type=strtobool,
            help="Use convolution module or not",
        )
        group.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
        )
        # Attention
        group.add_argument(
            "--adim",
            default=320,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        group.add_argument(
            "--zero-triu",
            default=False,
            type=strtobool,
            help="If true, zero the uppper triangular part of attention matrix.",
        )
        # Relative positional encoding
        group.add_argument(
            "--rel-pos-type",
            type=str,
            default="legacy",
            choices=["legacy", "latest"],
            help="Whether to use the latest relative positional encoding or the legacy one."
            "The legacy relative positional encoding will be deprecated in the future."
            "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
        )
        # Decoder
        group.add_argument("--dlayers", default=1, type=int, help="Number of decoder layers")
        group.add_argument("--dunits", default=320, type=int, help="Number of decoder hidden units")
        # -- pretrain
        group.add_argument("--pretrain-dataset", default="", type=str, help="pre-trained dataset for encoder")
        # -- custom name
        group.add_argument("--custom-pretrain-name", default="", type=str, help="pre-trained model for encoder")
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, odim, args, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # Check the relative positional encoding type
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        if self.rel_pos_type is None and args.transformer_encoder_attn_layer_type == "rel_mha":
            args.transformer_encoder_attn_layer_type = "legacy_rel_mha"
            logging.warning("Using legacy_rel_pos and it will be deprecated in the future.")

        idim = 80

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
        )

        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
        else:
            self.decoder = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x, extract_resnet_feats=False):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = x.unsqueeze(0)
        if extract_resnet_feats:
            resnet_feats = self.encoder(
                x,
                None,
                extract_resnet_feats=extract_resnet_feats,
            )
            return resnet_feats.squeeze(0)
        else:
            enc_output, _ = self.encoder(x, None)
            return enc_output.squeeze(0)


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    # Check for too much bias in height and width
    # if abs(center_y - img.shape[0] / 2) > height + threshold:
    #     raise Exception("too much bias in height")
    # if abs(center_x - img.shape[1] / 2) > width + threshold:
    #     raise Exception("too much bias in width")
    # Calculate bounding box coordinates
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    # Cut the image
    cutted_img = np.copy(img[y_min:y_max, x_min:x_max])
    return cutted_img


class VideoProcess:
    def __init__(
        self,
        crop_width=96,
        crop_height=96,
        start_idx=3,
        stop_idx=4,
        window_margin=12,
        convert_gray=True,
    ):
        self.reference = np.load("ckpt/avsr/20words_mean_face.npy")
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray

    def __call__(self, video, landmarks):
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames
        if not preprocessed_landmarks:
            return
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None
        return sequence

    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            window_margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean(
                [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)],
                axis=0,
            )
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray
            )
            patch = cut_patch(
                transformed_frame,
                transformed_landmarks[self.start_idx : self.stop_idx],
                self.crop_height // 2,
                self.crop_width // 2,
            )
            sequence.append(patch)
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks

    def affine_transform(
        self,
        frame,
        landmarks,
        reference,
        grayscale=False,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=(0, 1, 2, 3),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(reference, reference_size, target_size)
        transform = self.estimate_affine_transform(landmarks, stable_points, stable_reference)
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame,
            landmarks,
            transform,
            target_size,
            interpolation,
            border_mode,
            border_value,
        )

        return transformed_frame, transformed_landmarks

    def get_stable_reference(self, reference, reference_size, target_size):
        # -- right eye, left eye, nose tip, mouth center
        stable_reference = np.vstack(
            [
                np.mean(reference[36:42], axis=0),
                np.mean(reference[42:48], axis=0),
                np.mean(reference[31:36], axis=0),
                np.mean(reference[48:68], axis=0),
            ]
        )
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference

    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        return cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in stable_points]),
            stable_reference,
            method=cv2.LMEDS,
        )[0]

    def apply_affine_transform(
        self,
        frame,
        landmarks,
        transform,
        target_size,
        interpolation,
        border_mode,
        border_value,
    ):
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
        return transformed_frame, transformed_landmarks


class LandmarksDetector:
    def __init__(self, start_pts=0.0, end_pts=None):
        self.start_pts = start_pts
        self.end_pts = end_pts
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0
        )
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, filename):
        video_frames = torchvision.io.read_video(
            filename, start_pts=self.start_pts, end_pts=self.end_pts, pts_unit="sec"
        )[0].numpy()
        landmarks = self.detect(video_frames, self.full_range_detector)
        if all(element is None for element in landmarks):
            landmarks = self.detect(video_frames, self.short_range_detector)
            # assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].y
                            * ih
                        ),
                    ],
                ]
                face_points.append(lmx)
            # landmarks.append(np.array(face_points[max_id]))
            landmarks.append(np.array(face_points))
        return landmarks


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class VideoTransform:
    def __init__(self, speed_rate):
        self.video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x.unsqueeze(-1)),
            FunctionalModule(
                lambda x: (
                    x
                    if speed_rate == 1
                    else torch.index_select(
                        x,
                        dim=0,
                        index=torch.linspace(0, x.shape[0] - 1, int(x.shape[0] / speed_rate), dtype=torch.int64),
                    )
                )
            ),
            FunctionalModule(lambda x: x.permute(3, 0, 1, 2)),
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, sample):
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self):
        self.audio_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=0)),
            FunctionalModule(lambda x: x.transpose(0, 1)),
        )

    def __call__(self, sample):
        return self.audio_pipeline(sample)


class AVSRDataLoader:

    def __init__(
        self,
        modality,
        speed_rate=1,
        transform=True,
        convert_gray=True,
        start_pts=0.0,
        end_pts=None,
        frame_offset=0,
        num_frames=-1,
    ):
        self.modality = modality
        self.start_pts = start_pts
        self.end_pts = end_pts
        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.transform = transform
        if self.modality == "audio":
            self.audio_transform = AudioTransform()
        if self.modality == "video":
            self.video_process = VideoProcess(convert_gray=convert_gray)
            self.video_transform = VideoTransform(speed_rate=speed_rate)

    def load_data(self, data_filename, landmarks=None):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio, metadata = self.audio_process(audio, sample_rate)
            return self.audio_transform(audio) if self.transform else audio, metadata
        if self.modality == "video":
            video, metadata = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            return self.video_transform(video) if self.transform else video, metadata

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(
            data_filename, normalize=True, frame_offset=self.frame_offset, num_frames=self.num_frames
        )
        return waveform, sample_rate

    def load_video(self, data_filename):
        video, _, metadata = torchvision.io.read_video(
            data_filename, start_pts=self.start_pts, end_pts=self.end_pts, pts_unit="sec"
        )
        if "video_fps" in metadata:
            video_fps = metadata["video_fps"]
        else:
            video_fps = None
        video_num_frames = video.shape[0]
        return video.numpy(), {"video_fps": video_fps, "video_num_frames": video_num_frames}

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        audio_init_size = list(waveform.shape)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        audio_init_sample_rate = sample_rate
        audio_fin_size = list(waveform.shape)
        audio_fin_sample_rate = target_sample_rate
        return waveform, {
            "audio_init_size": audio_init_size,
            "audio_init_sample_rate": audio_init_sample_rate,
            "audio_fin_size": audio_fin_size,
            "audio_fin_sample_rate": audio_fin_sample_rate,
        }


class AVSR(torch.nn.Module):
    def __init__(
        self,
        model_path,
        model_conf,
        device="cuda",
    ):
        super(AVSR, self).__init__()
        self.device = device

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        labels_type = getattr(self.train_args, "labels_type", "char")
        if labels_type == "char":
            self.token_list = self.train_args.char_list
        elif labels_type == "unigram5000":
            file_path = "ckpt/avsr/unigram5000_units.txt"
            self.token_list = (
                ["<blank>"] + [word.split()[0] for word in open(file_path).read().splitlines()] + ["<eos>"]
            )
        self.odim = len(self.token_list)

        self.model = E2E(self.odim, self.train_args)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.to(device=self.device).eval()

    def forward(self, data):
        with torch.no_grad():
            features = self.model.encode(data.to(self.device))
        return features


def extract_data(pipeline_audio, pipeline_video, audio, video):
    audio_features = pipeline_audio(data=audio)
    video_features = pipeline_video(data=video)
    data = {
        "audio_features": audio_features,  # .detach().cpu().numpy(),
        "video_features": video_features,  # .detach().cpu().numpy(),
    }
    return data


def get_pipeline_obj(modality, device):
    if modality == "audio":
        return AVSR(
            model_path="ckpt/avsr/LRS3_A_WER1.0/model.pth",
            model_conf="ckpt/avsr/LRS3_A_WER1.0/model.json",
            device=device,
        )
    elif modality == "video":
        return AVSR(
            model_path="ckpt/avsr/LRS3_V_WER19.1/model.pth",
            model_conf="ckpt/avsr/LRS3_V_WER19.1/model.json",
            device=device,
        )
    else:
        raise Exception(f"{modality} modality is not supported.")


def get_features(audio, video, device):
    pipeline_audio = get_pipeline_obj("audio", device)
    pipeline_video = get_pipeline_obj("video", device)
    data = extract_data(pipeline_audio, pipeline_video, audio, video)
    return data


def get_dimodif_model(device):
    json_path = "ckpt/tfl/tfl_avdeepfake1m_reduceonplateau_256_8_2_5_True_whole.json"
    ckpt_path = "ckpt/tfl/tfl_avdeepfake1m_reduceonplateau_256_8_2_5_True_whole.pth"
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
