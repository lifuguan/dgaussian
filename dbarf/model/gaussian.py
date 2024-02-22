import os
import math

import torch

from dbarf.model.feature_network import ResUNet
from dbarf.depth_pose_network import DepthPoseNet
from dbarf.loss.photometric_loss import MultiViewPhotometricDecayLoss

from dbarf.base.model_base import Model

from dbarf.model.pixelsplat.decoder import get_decoder
from dbarf.model.pixelsplat.encoder import get_encoder
from dbarf.model.pixelsplat.pixelsplat import PixelSplat
from dbarf.dataset.data_module import get_data_shim

class GaussianModel(Model):
    def __init__(self, args, load_opt=True, load_scheduler=True, pretrained=True):
        self.args = args
        device = torch.device(f'cuda:{args.local_rank}')
        
        # create generalized 3d gaussian.
        encoder, encoder_visualizer = get_encoder(args.pixelsplat.encoder)
        decoder = get_decoder(args.pixelsplat.decoder)
        self.gaussian_model = PixelSplat(encoder, decoder, encoder_visualizer)
        self.gaussian_model.to(device)
        
        self.photometric_loss = MultiViewPhotometricDecayLoss()

    def to_distributed(self):
        # super().to_distributed()

        if self.args.distributed:
            self.gaussian_model = torch.nn.parallel.DistributedDataParallel(
                self.gaussian_model,
                device_ids=[self.args.local_rank],
                output_device=[self.args.local_rank]
            )

    def switch_to_eval(self):

        self.gaussian_model.eval()

    def switch_to_train(self):
        self.gaussian_model.train()

    def switch_state_machine(self, state='joint') -> str:
        if state == 'pose_only':
            self._set_gaussian_state(opt=False)
        
        elif state == 'nerf_only':
            self._set_gaussian_state(opt=True)
        
        elif state == 'joint':
            self._set_gaussian_state(opt=True)
        
        else:
            raise NotImplementedError("Not supported state")
        
        return state

    def _set_gaussian_state(self, opt=True):
        for param in self.gaussian_model.parameters():
            param.requires_grad = opt
    
    def compose_joint_loss(self, sfm_loss, nerf_loss, step, coefficient=1e-5):
        # The jointly training loss is composed by the convex_combination:
        #   L = a * L1 + (1-a) * L2
        alpha = math.pow(2.0, -coefficient * step)
        loss = alpha * sfm_loss + (1 - alpha) * nerf_loss
        
        return loss
