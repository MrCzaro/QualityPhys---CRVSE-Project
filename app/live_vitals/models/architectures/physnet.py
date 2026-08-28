"""PhysNet 3D-CNN encoder-decoder (Yu et al. 2019).

Architecture is fixed by the Phase-3 training run (NB_P3_18); the published
checkpoint loads against it with strict=True.
"""
import torch.nn as nn

from ...config import CLIP_LEN


class PhysNet(nn.Module):
    """Maps a normalized face-crop clip [B,3,T,H,W] to a BVP waveform [B,T]."""

    def __init__(self, frames=CLIP_LEN):
        super().__init__()
        self.frames = frames

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, (3, 3, 3), stride=1, padding=1),
                nn.BatchNorm3d(cout), nn.ReLU(inplace=True))

        self.b1 = nn.Sequential(
            nn.Conv3d(3, 16, (1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.b2 = block(16, 32); self.b3 = block(32, 64)
        self.b4 = block(64, 64); self.b5 = block(64, 64)
        self.b6 = block(64, 64); self.b7 = block(64, 64)
        self.b8 = block(64, 64); self.b9 = block(64, 64)
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64), nn.ELU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64), nn.ELU(inplace=True))
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.out = nn.Conv3d(64, 1, (1, 1, 1), stride=1, padding=0)
        self.maxpool_spa = nn.MaxPool3d((1, 2, 2))
        self.maxpool_spatem = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        x = self.b1(x); x = self.maxpool_spa(x)
        x = self.b2(x); x = self.b3(x); x = self.maxpool_spatem(x)
        x = self.b4(x); x = self.b5(x); x = self.maxpool_spatem(x)
        x = self.b6(x); x = self.b7(x); x = self.maxpool_spa(x)
        x = self.b8(x); x = self.b9(x)
        x = self.up1(x); x = self.up2(x)
        x = self.poolspa(x); x = self.out(x)
        return x.view(x.shape[0], self.frames)