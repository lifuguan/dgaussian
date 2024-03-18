from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn, optim

from .types import Gaussians
from ...dataset.data_module import get_data_shim
from ...dataset.types import BatchedExample
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
import math
from torchvision.utils import save_image
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

    def setup_optimizer(self):
        # self.optimizer = torch.optim.Adam([
        #     dict(params=self.model.gaussian_model.parameters(),lr=self.config.lrate_mlp)
        # ])

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                  step_size=self.config.lrate_decay_steps,
        #                                                  gamma=self.config.lrate_decay_factor)

        self.optimizer = torch.optim.Adam(self.model.gaussian_model.parameters(), lr=self.config.optimizer.lr)
        warm_up_steps = self.config.optimizer.warm_up_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                        1 / warm_up_steps,
                                                        1,
                                                        total_iters=warm_up_steps)



    def forward(self, batch, global_step: int,i:int = 3,j:int = 3):  #默认进全图

        features=None
        _, _, _, h, w = batch["target"]["image"].shape
        if i!=3 and j!=3:  #进行crop
            features = self.encoder(batch["context"], global_step,None,4,4) #五张图先进去算出feaure

            # Run the model.
            for k in range(batch["context"]["image"].shape[1] - 1):
                tmp_batch = self.batch_cut(batch["context"],k)
                tmp_gaussians = self.encoder(tmp_batch, global_step,features[:,k:k+2,:,:,:],i,j,True) #默认进全图即i=3，j=3
                if k == 0:
                    gaussians: Gaussians = tmp_gaussians
                else:
                    gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
                    gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
                    gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
                    gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
            # if gaussian_last!=None:
            #     gaussians.covariances = torch.cat([gaussians.covariances, gaussian_last.covariances], dim=1)
            #     gaussians.means = torch.cat([gaussians.means, gaussian_last.means], dim=1)
            #     gaussians.harmonics = torch.cat([gaussians.harmonics, gaussian_last.harmonics], dim=1)
            #     gaussians.opacities = torch.cat([gaussians.opacities, gaussian_last.opacities], dim=1)
            # gaussians = gaussians.to(torch.bfloat16)
            output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode='depth'
            )
        
            
            ret = {'rgb': output.color, 'depth': output.depth}
            target_gt = {'rgb': batch["target"]["image"]}
            return ret, target_gt
        else:


            
            # Run the model.

            # batch["target"]["intrinsics"][:,:,0,0]=batch['target']['intrinsics'][:,:,0,0]*0.5
            # batch['target']['intrinsics'][:,:,1,1]=batch['target']['intrinsics'][:,:,1,1]*0.5

            # translation = torch.tensor([3, 0, 0]).cuda().float().unsqueeze(-1)
            # # batch["target"]["extrinsics"][0][0][:, 3] += torch.mm(batch["target"]["extrinsics"][0][0][:, :3], translation) .squeeze(-1)   

            # # 将角度从度转换为弧度
            # theta = 10 * math.pi / 180

            # # 构造绕 Z 轴旋转的旋转矩阵
            # rotation_matrix = torch.tensor([[math.cos(theta), -math.sin(theta), 0, 0],
            #                                 [math.sin(theta),  math.cos(theta), 0, 0],
            #                                 [0,               0,               1, 0],
            #                                 [0,               0,               0, 1]], device='cuda:0')

            # # 应用旋转
            # batch["target"]["extrinsics"][0][0] = torch.matmul(rotation_matrix, batch["target"]["extrinsics"][0][0])





            features = self.encoder(batch["context"], global_step,None,4,4)
            for k in range(batch["context"]["image"].shape[1] - 1):
                tmp_batch = self.batch_cut(batch["context"],k)
                tmp_gaussians = self.encoder(tmp_batch, global_step,features[:,k:k+2,:,:,:],i,j,True) #默认进全图即i=3，j=3
                if k == 0:
                    gaussians: Gaussians = tmp_gaussians
                else:
                    gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
                    gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
                    gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
                    gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
        # gaussians = self.encoder(batch['context'], global_step, False)
            
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode='depth'
            )

            ret = {'rgb': output.color, 'depth': output.depth}
            save_image(output.color[0][0],f'render{52}.png')
            target_gt = {'rgb': batch["target"]["image"]}
            return ret, target_gt
    
    def batch_cut(self, batch, i):
        return {
            'extrinsics': batch['extrinsics'][:,i:i+2,:,:],
            'intrinsics': batch['intrinsics'][:,i:i+2,:,:],
            'image': batch['image'][:,i:i+2,:,:,:],
            'near': batch['near'][:,i:i+2],
            'far': batch['far'][:,i:i+2],
            # 'index': batch['index'][:,i:i+2],
        }