from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn, optim

from ...dataset.data_module import get_data_shim
from ...dataset.types import BatchedExample
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class PixelSplat(nn.Module):
    encoder: nn.Module
    decoder: Decoder

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        encoder_visualizer: Optional[EncoderVisualizer],
    ) -> None:
        super().__init__()

        # Set up the model.
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_visualizer = encoder_visualizer
        
        self.data_shim = get_data_shim(self.encoder)

    def forward(self, batch_, global_step: int):
        batch: BatchedExample = self.data_shim(batch_)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians = self.encoder(batch["context"], global_step, False)
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode='depth'
        )
        target_gt = batch["target"]["image"]
        
        return output, target_gt