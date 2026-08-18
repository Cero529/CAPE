import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class AVDeepFakeDetector(nn.Module):

    def __init__(
        self,
        task: str,
        max_length: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        win_size: int = 0,
        dropout: float = 0.5,
        feature_pyramid: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.task = task
        self.max_length = max_length
        self.d_model = d_model
        self.win_size = win_size
        self.feature_pyramid = feature_pyramid
        self.device = device
        self.num_seq = 2
        if self.task == "dfd":
            self.cls = nn.Embedding(1, d_model)
        self.sep = nn.Embedding(1, d_model)
        self.seq_encoder = nn.Embedding(self.num_seq, d_model)
        self.npositions = self.num_seq * max_length + self.num_seq - int(self.task == "tfl")
        self.pos_encoder = nn.Embedding(self.npositions, d_model)
        self.projection = nn.Sequential(
            nn.Conv1d(768, d_model, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.construct_mask()
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                nhead,
                d_hid,
                dropout,
                batch_first=True,
            ),
            nlayers,
        )
        self.hooks = [
            Hook(name, module) for name, module in self.transformer_encoder.named_modules() if "norm2" in name
        ]
        if self.task == "dfd":
            self.classification = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, 2),
            )
        else:
            self.classification = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding="same"),
                nn.LayerNorm([d_model, 2 * self.max_length]),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding="same"),
                nn.LayerNorm([d_model, 2 * self.max_length]),
                nn.ReLU(),
                nn.Conv1d(d_model, 1, kernel_size=3, padding="same"),
            )
            self.regression = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding="same"),
                nn.LayerNorm([d_model, 2 * self.max_length]),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding="same"),
                nn.LayerNorm([d_model, 2 * self.max_length]),
                nn.ReLU(),
                nn.Conv1d(d_model, 2, kernel_size=3, padding="same"),
                nn.ReLU(),
            )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.task == "dfd":
            self.cls.weight.data.uniform_(-initrange, initrange)
        self.sep.weight.data.uniform_(-initrange, initrange)
        self.pos_encoder.weight.data.uniform_(-initrange, initrange)
        self.seq_encoder.weight.data.uniform_(-initrange, initrange)
        self.projection[0].weight.data.uniform_(-initrange, initrange)
        self.projection[0].bias.data.uniform_(-initrange, initrange)
        self.projection[2].weight.data.uniform_(-initrange, initrange)
        self.projection[2].bias.data.uniform_(-initrange, initrange)

    def construct_mask(self):
        if self.win_size != 0:
            seq_len = self.max_length + self.win_size
            self.mask = torch.stack(
                [
                    torch.cat(
                        (torch.zeros((i,)), torch.ones((self.win_size,)), torch.zeros(seq_len - self.win_size - i))
                    )
                    for i in range(self.max_length)
                ],
                dim=0,
            )[:, self.win_size // 2 : -self.win_size // 2]
            self.mask = torch.cat((self.mask, torch.ones((self.mask.shape[0], 1)), self.mask), dim=1)
            self.mask = torch.cat((self.mask, torch.ones((1, self.mask.shape[1])), self.mask), dim=0)
            if self.task == "dfd":
                self.mask = torch.cat((torch.ones((self.mask.shape[0], 1)), self.mask), dim=1)
                self.mask = torch.cat((torch.ones((1, self.mask.shape[1])), self.mask), dim=0)
            self.mask = torch.logical_not(self.mask.bool()).to(self.device)
        else:
            self.mask = None

    def forward(self, src):
        zero_idx = torch.zeros(src[0].shape[0], dtype=torch.int).to(self.device)
        one_idx = torch.ones(src[0].shape[0], dtype=torch.int).to(self.device)
        pos_idx = torch.arange(self.npositions).to(self.device)

        video = self.projection(src[0].permute((0, 2, 1))).permute((0, 2, 1))
        video = video + self.seq_encoder(zero_idx).unsqueeze(1)
        if self.task == "dfd":
            video = torch.cat((self.cls(zero_idx).unsqueeze(1), video), dim=1)

        audio = self.projection(src[1].permute((0, 2, 1))).permute((0, 2, 1))
        audio = audio + self.seq_encoder(one_idx).unsqueeze(1)
        audio = torch.cat((self.sep(zero_idx).unsqueeze(1), audio), dim=1)

        input = torch.cat((video, audio), dim=1)
        input += self.pos_encoder(pos_idx).unsqueeze(0)
        if self.win_size != 0:
            z = self.transformer_encoder(input, mask=self.mask)
        else:
            z = self.transformer_encoder(input)

        if self.task == "dfd":
            if self.feature_pyramid:
                z = torch.stack([h.output for h in self.hooks], dim=1)
                z = z[:, :, 0, :]
                p = self.classification(z)
            else:
                z = z[:, 0, :]
                p = self.classification(z)
        else:
            if self.feature_pyramid:
                z = torch.stack([h.output for h in self.hooks], dim=1)
                z = torch.cat((z[:, :, : self.max_length, :], z[:, :, self.max_length + 1 :, :]), dim=2)
                z_perm = z.permute((0, 1, 3, 2)).reshape((-1, self.d_model, 2 * self.max_length))
                p = (
                    torch.cat((self.classification(z_perm), self.regression(z_perm)), dim=1)
                    .reshape((z.shape[0], z.shape[1], 3, 2 * self.max_length))
                    .permute((0, 1, 3, 2))
                )
            else:
                z = torch.cat((z[:, : self.max_length, :], z[:, self.max_length + 1 :, :]), dim=1)
                z_perm = z.permute((0, 2, 1))
                p = torch.cat((self.classification(z_perm), self.regression(z_perm)), dim=1).permute((0, 2, 1))
        return p, z
