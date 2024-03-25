import random
import numpy as np

import hydra
from omegaconf import DictConfig

import torch
import torch.utils.data.distributed

from dbarf.geometry.depth import inv2depth
from dbarf.geometry.align_poses import align_ate_c2b_use_a2b
from dbarf.model.dbarf import DBARFModel
from dbarf.model.dgaussian import DGaussianModel
from dbarf.projection import Projector
from dbarf.pose_util import Pose, rotation_distance
from dbarf.render_ray import render_rays
from dbarf.render_image import render_single_image
from dbarf.sample_ray import RaySamplerSingleImage
from dbarf.visualization.pose_visualizer import visualize_cameras
from dbarf.visualization.feature_visualizer import *
from utils_loc import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr, data_shim
from train_ibrnet import synchronize
from dbarf.base.trainer import BaseTrainer

import dbarf.config as config
from dbarf.loss.criterion import MaskedL2ImageLoss, self_sup_depth_loss
from math import ceil
import copy
# torch.autograd.set_detect_anomaly(True)

def random_crop(data,size=[160,224] ,center=None):
    _,_,_,h, w = data['context']['image'].shape
    # size=torch.from_numpy(size)
    batch=copy.deepcopy(data)
    out_h, out_w = size[0], size[1]

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)
    # batch['context']['image'] = batch['context']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]
    # batch['target']['image'] = batch['target']['image'][:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]

    # batch['context']['intrinsics'][:,:,0,0]=batch['context']['intrinsics'][:,:,0,0]*w/out_w
    # batch['context']['intrinsics'][:,:,1,1]=batch['context']['intrinsics'][:,:,1,1]*h/out_h
    # batch['context']['intrinsics'][:,:,0,2]=(batch['context']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    # batch['context']['intrinsics'][:,:,1,2]=(batch['context']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h

    # batch['target']['intrinsics'][:,:,0,0]=batch['target']['intrinsics'][:,:,0,0]*w/out_w
    # batch['target']['intrinsics'][:,:,1,1]=batch['target']['intrinsics'][:,:,1,1]*h/out_h
    # batch['target']['intrinsics'][:,:,0,2]=(batch['target']['intrinsics'][:,:,0,2]*w-center_w+out_w // 2)/out_w
    # batch['target']['intrinsics'][:,:,1,2]=(batch['target']['intrinsics'][:,:,1,2]*h-center_h+out_h // 2)/out_h
    return batch,center_h,center_w
class DGaussianTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.state = 'pose_only'
        self.projector = Projector(device=self.device)

    def build_networks(self):
        self.model = DGaussianModel(self.config,
                                load_opt=not self.config.no_load_opt,
                                load_scheduler=not self.config.no_load_scheduler,
                                pretrained=self.config.pretrained)

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.gaussian_model.parameters(), lr=self.config.optimizer.lr)
        warm_up_steps = self.config.optimizer.warm_up_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                        1 / warm_up_steps,
                                                        1,
                                                        total_iters=warm_up_steps)
        self.pose_optimizer = torch.optim.Adam([
            dict(params=self.model.pose_learner.parameters(), lr=self.config.lrate_pose)
        ])
        self.pose_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pose_optimizer, step_size=self.config.lrate_decay_pose_steps, gamma=0.5)

    def setup_loss_functions(self):
        self.rgb_loss = MaskedL2ImageLoss()


    def compose_state_dicts(self) -> None:
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict()}
        self.state_dicts['models']['pose_learner'] = self.model.pose_learner
        self.state_dicts['models']['gaussian'] = self.model.gaussian_model
        
        self.state_dicts['optimizers']['pose_optimizer'] = self.pose_optimizer
        self.state_dicts['schedulers']['pose_scheduler'] = self.pose_scheduler

    def train_iteration(self, batch) -> None:
        ######################### 3-stages training #######################
        # ---- (1) Train the pose optimizer with self-supervised loss.<---|
        # |             (10000 iterations)                                |
        # |--> (2) Train ibrnet while fixing the pose optimizer.          |
        # |             (10000 iterations)                                |
        # |--> (3) Jointly train the pose optimizer and ibrnet.           |
        # |             (10000 iterations)                                |
        # |-------------------------->------------------------------------|
        # if self.iteration % 5000 == 0 and (self.iteration // 5000) % 2 == 0:
        #     self.state = self.model.switch_state_machine(state='pose_only')
        # elif self.iteration % 5000 == 0 and (self.iteration // 5000) % 2 == 1:
        #     self.state = self.model.switch_state_machine(state='nerf_only')
        # if self.iteration != 0 and self.iteration % 10000 == 0:
        #     self.state = self.model.switch_state_machine(state='joint')
        if self.iteration == 0:
            self.state = self.model.switch_state_machine(state='joint')

        min_depth, max_depth = batch['depth_range'][0][0], batch['depth_range'][0][1]
        coefficient=1e-5
        alpha = math.pow(2.0, -coefficient * self.iteration)
        # Start of core optimization loop
        pred_inv_depths, pred_rel_poses, sfm_loss, fmap = self.model.correct_poses(
            fmaps=None,
            target_image=batch['rgb'].cuda(),
            ref_imgs=batch['src_rgbs'].cuda(),
            target_camera=batch['camera'],
            ref_cameras=batch['src_cameras'],
            min_depth=min_depth,
            max_depth=max_depth,
            scaled_shape=batch['scaled_shape'])

        # The predicted inverse depth is used as a weak supervision to NeRF.
        self.pred_inv_depth = pred_inv_depths[-1]
        inv_depth_prior = pred_inv_depths[-1].detach().clone()
        inv_depth_prior = inv_depth_prior.reshape(-1, 1)

        if self.config.use_pred_pose is True:
            num_views = batch['src_cameras'].shape[1]
            target_pose = batch['camera'][0,-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1).to(self.device)
            context_poses = self.projector.get_train_poses(target_pose, pred_rel_poses[:, -1, :])
            batch['context']['extrinsics'] = context_poses.unsqueeze(0).detach()
        
        batch = data_shim(batch, device=self.device)

        self.optimizer.zero_grad()
        self.pose_optimizer.zero_grad()
        with torch.no_grad():
            batch = self.model.gaussian_model.data_shim(batch)
            ret, data_gt = self.model.gaussian_model(batch, self.iteration)
        ret['rgb'].requires_grad_(True)
        # ret_ref['rgb'].requires_grad_(True)
        coarse_loss = self.rgb_loss(ret, data_gt)
        # coarse_loss_ref = self.rgb_loss(ret_ref, data_gt_ref)
        a=0.1
        # coarse_loss=coarse_loss+ a*coarse_loss_ref
        coarse_loss.backward()     
        rgb_pred_grad=ret['rgb'].grad
        # rgb_pred_grad_ref=ret_ref['rgb'].grad
        # rgb_pred_grad =torch.cat([rgb_pred_grad,rgb_pred_grad_ref],dim=1)
        _, _, _, h, w = batch["target"]["image"].shape
        out_h=176
        out_w=240
        row=ceil(h/out_h)
        col=ceil(w/out_w)
        for i in range(row):
            for j in range(col):
                if i==row-1 and j==col-1:
                    data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(h-out_h//2),int(w-out_w//2)))
                elif i==row-1:#最后一行
                    data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(h-out_h//2),int(out_w//2+j*out_w)))
                elif j==col-1:#z最后一列
                    data_crop,center_h,center_w=random_crop( batch,size=[out_h,out_w],center=(int(out_h//2+i*out_h),int(w-out_w//2)))
                else:
                    data_crop,center_h,center_w=random_crop(batch,size=[out_h,out_w],center=(int(out_h//2+i*out_h),int(out_w//2+j*out_w)))  
                # Run the model.
                if i==0 and j==0:
                    ret_patch, data_gt_patch = self.model.gaussian_model(data_crop, self.iteration,i,j)
                else:
                    ret_patch, data_gt_patch= self.model.gaussian_model(data_crop, self.iteration,i,j)
        # coarse_loss = self.rgb_loss(ret_patch, data_gt_patch)
        # coarse_loss.backward()
                ret_patch['rgb']=ret_patch['rgb'][:,:,:,out_h*i : out_h*(i+1), out_w*j : out_w*(j+1)]
                # ret_patch_ref['rgb']=ret_patch_ref['rgb'][:,:,:,out_h*i : out_h*(i+1),out_w*j : out_w*(j+1)]
                # mask = torch.zeros_like(ret_patch['rgb'])
                # mask[:,:,:,center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2]=1
                # mask = mask.to(self.device)
                # coarse_loss = self.rgb_loss(ret_patch, data_gt_patch*mask)
                # coarse_loss.backward()
                # ret_patch['rgb'] = torch.cat([ret_patch['rgb'],ret_patch_ref['rgb']],dim=1)
                ret_patch['rgb'].backward((1-alpha)*rgb_pred_grad[:,:,:,out_h*i : out_h*(i+1), out_w*j : out_w*(j+1)])
        # print(ret_patch['rgb'].grad)
        # loss_all = 0
        # loss_dict = {}
        sfm_loss=alpha*sfm_loss['loss']
        sfm_loss.backward()
        # coarse_loss = self.rgb_loss(ret, data_gt)
        # loss_dict['gaussian_loss'] = coarse_loss

        # if self.config.use_depth_loss is True:
        #     rendered_depth = ret['depth'][0].permute(1, 2, 0).reshape(-1, 1)
        #     loss_depth = self_sup_depth_loss(inv_depth_prior, rendered_depth, min_depth, max_depth)
        #     loss_dict['self-sup-depth'] = loss_depth

        # if self.state == 'pose_only' or self.state == 'joint':
        #     loss_dict['sfm_loss'] = sfm_loss['loss']
        #     self.scalars_to_log['loss/photometric_loss'] = sfm_loss['metrics']['photometric_loss']
        #     if 'smoothness_loss' in sfm_loss['metrics']:
        #         self.scalars_to_log['loss/smoothness_loss'] = sfm_loss['metrics']['smoothness_loss']

        # if self.state == 'joint':
        #     if self.config.use_depth_loss is True:
        #         loss_all += loss_dict['self-sup-depth'].item() * 0.04
        #     loss_all += self.model.compose_joint_loss(
        #         loss_dict['sfm_loss'], loss_dict['gaussian_loss'], self.iteration)
        # elif self.state == 'pose_only':
        #     loss_all += loss_dict['sfm_loss']
        # else: # nerf_only
        #     loss_all += loss_dict['self-sup-depth'].item() * 0.04
        #     loss_all += loss_dict['gaussian_loss']

        # # with torch.autograd.detect_anomaly():
        # loss_all.backward()
        if self.state == 'pose_only' or self.state == 'joint':
            self.pose_optimizer.step()
            self.pose_scheduler.step()

        if self.state == 'nerf_only' or self.state == 'joint':
            self.optimizer.step()
            self.scheduler.step()

        if self.config.local_rank == 0 and self.iteration % self.config.n_tensorboard == 0:
            mse_error = img2mse(ret['rgb'][0], data_gt['rgb'][0]).item()
            self.scalars_to_log['train/coarse-loss'] = mse_error
            self.scalars_to_log['train/coarse-psnr'] = mse2psnr(mse_error)
            # self.scalars_to_log['loss/final'] = loss_all.item()
            self.scalars_to_log['loss/rgb_coarse'] = coarse_loss.detach().item()
            # print(f"corse loss: {mse_error}, psnr: {mse2psnr(mse_error)}")
            self.scalars_to_log['lr/Gaussian'] = self.scheduler.get_last_lr()[0]
            self.scalars_to_log['lr/pose'] = self.pose_scheduler.get_last_lr()[0]
            
            aligned_pred_poses, poses_gt = align_predicted_training_poses(
                pred_rel_poses[:, -1, :], self.train_data, self.train_dataset, self.config.local_rank)
            pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
            # visualize_cameras(self.visdom, step=self.iteration, poses=[aligned_pred_poses, poses_gt], cam_depth=0.1)

            self.scalars_to_log['train/R_error_mean'] = pose_error['R_error_mean']
            self.scalars_to_log['train/t_error_mean'] = pose_error['t_error_mean']
            self.scalars_to_log['train/R_error_med'] = pose_error['R_error_med']
            self.scalars_to_log['train/t_error_med'] = pose_error['t_error_med']

    def validate(self) -> float:
        self.model.switch_to_eval()

        target_image = self.train_data['rgb'].squeeze(0).permute(2, 0, 1)
        pred_inv_depth_gray = self.pred_inv_depth.squeeze(0).detach().cpu()
        pred_inv_depth = self.pred_inv_depth.squeeze(0).squeeze(0)
        pred_depth= inv2depth(pred_inv_depth)
        pred_depth_color = colorize(pred_depth.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)

        self.writer.add_image('train/target_image', target_image, self.iteration)
        self.writer.add_image('train/pred_inv_depth', pred_inv_depth_gray, self.iteration)
        self.writer.add_image('train/pred_depth-color', pred_depth_color, self.iteration)

        # Logging a random validation view.
        val_data = next(self.val_loader_iterator)
        score = log_view_to_tb(
            self.writer, self.iteration, self.config, self.model,
            render_stride=self.config.render_stride, prefix='val/',
            data=val_data, dataset=self.val_dataset, device=self.device)
        torch.cuda.empty_cache()
        self.model.switch_to_train()

        return score


@torch.no_grad()
def get_predicted_training_poses(pred_poses):
    target_pose = torch.eye(4, device=pred_poses.device, dtype=torch.float).repeat(1, 1, 1)

    # World->camera poses.
    pred_poses = Pose.from_vec(pred_poses) # [n_views, 4, 4]
    pred_poses = torch.cat([target_pose, pred_poses], dim=0)

    # Convert camera poses to camera->world.
    pred_poses = pred_poses.inverse()

    return pred_poses


@torch.no_grad()
def align_predicted_training_poses(pred_poses, data, dataset, device):
    target_pose_gt = data['camera'][..., -16:].reshape(1, 4, 4)
    src_poses_gt = data['src_cameras'][..., -16:].reshape(-1, 4, 4)
    poses_gt = torch.cat([target_pose_gt, src_poses_gt], dim=0).to(device).float()
    
    pred_poses = get_predicted_training_poses(pred_poses)

    aligned_pred_poses = align_ate_c2b_use_a2b(pred_poses, poses_gt)

    return aligned_pred_poses, poses_gt


@torch.no_grad()
def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    # measure errors in rotation and translation
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)
    
    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)
    
    mean_rotation_error = np.rad2deg(R_error.mean().cpu())
    mean_position_error = t_error.mean()
    med_rotation_error = np.rad2deg(R_error.median().cpu())
    med_position_error = t_error.median()
    
    return {'R_error_mean': mean_rotation_error, "t_error_mean": mean_position_error,
            'R_error_med': med_rotation_error, 't_error_med': med_position_error}


@torch.no_grad()
def log_view_to_tb(writer, global_step, args, model, render_stride=1, prefix='', data=None, dataset=None, device=None) -> float:

    pred_inv_depth, pred_rel_poses, _, __ = model.correct_poses(
                            fmaps=None,
                            target_image=data['rgb'].cuda(),
                            ref_imgs=data['src_rgbs'].cuda(),
                            target_camera=data['camera'].cuda(),
                            ref_cameras=data['src_cameras'].cuda(),
                            min_depth=data['depth_range'][0][0],
                            max_depth=data['depth_range'][0][1],
                            scaled_shape=data['scaled_shape'])
    inv_depth_prior = pred_inv_depth.reshape(-1, 1).detach().clone()

    if prefix == 'val/':
        pred_inv_depth = pred_inv_depth.squeeze(0).squeeze(0)
        pred_inv_depth = colorize(pred_inv_depth.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)
        writer.add_image(prefix + 'pred_inv_depth', pred_inv_depth, global_step)
        aligned_pred_poses, poses_gt = align_predicted_training_poses(pred_rel_poses, data, dataset, args.local_rank)
        pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
        writer.add_scalar('val/R_error_mean', pose_error['R_error_mean'], global_step)
        writer.add_scalar('val/t_error_mean', pose_error['t_error_mean'], global_step)
        writer.add_scalar('val/R_error_med', pose_error['R_error_med'], global_step)
        writer.add_scalar('val/t_error_med', pose_error['t_error_med'], global_step)

    batch = data_shim(data, device=device)
    batch = model.gaussian_model.data_shim(batch)
    ret, data_gt = model.gaussian_model(batch, global_step)
        

    average_im = batch['context']['image'].cpu().mean(dim=(0, 1))


    rgb_gt = data_gt['rgb'][0][0]

    rgb_pred = ret['rgb'][0][0].detach().cpu()

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['depth'].detach().cpu()[0][0]


    depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_im_gt_pred', rgb_im, global_step)
    writer.add_image(prefix + 'depth_pred', depth_im, global_step)

    # plot_feature_map(writer, global_step, ray_sampler, feat_maps, prefix)

    # write scalar
    # pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(rgb_pred, data_gt['rgb'][0][0].detach().cpu())
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    return psnr_curr_img

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="finetune_dgaussian_stable",
)

def train(cfg_dict: DictConfig):
    args = cfg_dict
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Configuration for distributed training.
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        print(f'[INFO] Train in distributed mode')
        
    device = "cuda:{}".format(args.local_rank)

    trainer = DGaussianTrainer(args)
    trainer.train()

if __name__ == '__main__':
    train()
