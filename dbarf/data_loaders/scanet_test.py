import random
import time
import os
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import pandas as pd
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append("../")
# from .dataset import DatasetCfgCommon
from torch.utils.data import IterableDataset
from typing import Literal
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
from .base_utils import downsample_gaussian_blur
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
from torch.utils.data import Dataset


def loader_resize(rgb, camera, src_rgbs, src_cameras, size=(400, 600)):
    h, w = rgb.shape[:2]
    out_h, out_w = size[0], size[1]
    intrinsics = camera[2:18].reshape(4, 4)
    src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras, intrinsics[..., :3, :3], src_intrinsics[..., :3, :3]

    ratio_y = out_h / h
    ratio_x = out_w / w
    intrinsics[:1, :1] *= ratio_y
    src_intrinsics[:, :1, :1] *= ratio_y
    intrinsics[1:2, 1:2] *= ratio_x
    src_intrinsics[:, 1:2, 1:2] *= ratio_x
    camera[0] = out_h
    camera[1] = out_w
    camera[2:18] = intrinsics.flatten()
    src_cameras[:, 0] = out_h
    src_cameras[:, 1] = out_w
    src_cameras[:, 2:18] = src_intrinsics.reshape(-1, 16)
    rgb = cv2.resize(downsample_gaussian_blur(
                rgb, ratio_y), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    src_rgbs = [cv2.resize(downsample_gaussian_blur(
                src_rgb, ratio_y), (out_w, out_h), interpolation=cv2.INTER_LINEAR) for src_rgb in src_rgbs]
    src_rgbs = np.stack(src_rgbs, axis=0)
    return rgb, camera, src_rgbs, src_cameras, intrinsics[..., :3, :3], src_intrinsics[..., :3, :3]

class  PointSegClassMapping(object):
    """Map original semantic class to valid category ids.
    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        # neg_cls = -1

        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, seg_label):
        """Call function to map original semantic class to valid category ids.
        Args:
            results (dict): Result dict containing point semantic masks.
        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.
                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        seg_label = np.clip(seg_label, 0, self.max_cat_id)
        return self.cat_id2class[seg_label]

def set_seed(index,mode):
    if mode == "train":
        np.random.seed((index+int(time.time()))%(2**16))
        random.seed((index+int(time.time()))%(2**16)+1)
        torch.random.manual_seed((index+int(time.time()))%(2**16)+1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)

# only for training
class ScannetTrainDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        # if mode == "train":
        #     self.scene_path_list = np.loadtxt('configs/scannetv2_train_split.txt',dtype=str).tolist()
        # elif mode == "val":
        #     self.scene_path_list = np.loadtxt('configs/scannetv2_val_split.txt',dtype=str).tolist()
        # else:
        #     self.scene_path_list = np.loadtxt('configs/scannetv2_test_split.txt',dtype=str).tolist()
        self.args = args
        self.mode = mode
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = False

        # image_size = 320
        # self.ratio = image_size / 1296
        # self.h, self.w = int(self.ratio*972), int(image_size)
        self.image_size = (224, 320-32)
        self.ratio = self.image_size[1] / 1296

        all_rgb_files, all_depth_files, all_pose_files, all_label_files, all_intrinsics_files = [],[],[],[],[]
        # for i, scene in enumerate(self.scene_path_list):
        scene_path = os.path.join('/data/gyy/lsplat/datasets/scannet/scene0000_00')
        pose_files = []
        i = 0
        for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
            path = os.path.join(scene_path, "pose", f'{i}.txt')
            pose = np.loadtxt(path)
            if np.isinf(pose).any() or np.isnan(pose).any():
                continue
            else:
                pose_files.append(path)
                i = i + 1
            if i == 200:
                break

        i_test = np.arange(len(pose_files))[::self.args.llffhold] if mode != 'eval_pose' else []
        i_train = np.array([j for j in np.arange(len(pose_files)) if
                                (j not in i_test and j not in i_test)])
        rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
        depth_files = [f.replace("pose", "depth").replace("txt", "png") for f in pose_files]
        intrinsics_files = [
            os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
        ]
        label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

        all_rgb_files.append(rgb_files)
        all_depth_files.append(depth_files)
        all_label_files.append(label_files)
        all_pose_files.append(pose_files)
        all_intrinsics_files.append(intrinsics_files)

        if mode == 'train' or mode == 'eval_pose':
            i_render = i_train
        else:
            i_render = i_test

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files, dtype=object)[index]
        self.all_depth_files = np.array(all_depth_files, dtype=object)[index][:,i_render]
        self.all_pose_files = np.array(all_pose_files, dtype=object)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files, dtype=object)[index]
        self.render_pose_files = np.array(all_pose_files, dtype=object)[index][:,i_render]
        self.render_rgb_files = np.array(all_rgb_files, dtype=object)[index][:,i_render]
    def pose_inverse(self, pose):
        R = pose[:, :3].T
        t = - R @ pose[:, 3:]
        inversed_pose = np.concatenate([R, t], -1)
        return np.concatenate([inversed_pose, [[0, 0, 0, 1]]])
        # return inversed_pose

    def __len__(self):
        return 9999  # 确保不会中断
    
    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)
    
    def normalize_intrinsics(self, intrinsics, img_size):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized
    
    def __getitem__(self, idx): 
            id_scene = idx
            # id_scene = 122
            real_idx = id_scene % len(self.all_rgb_files)
            real_idx = 0
            rgb_files = self.all_rgb_files[real_idx]
            render_rgb_files = self.render_rgb_files[real_idx]
            scene = rgb_files[0].split('/')[2]
            depth_files = self.all_depth_files[real_idx]
            pose_files = self.all_pose_files[real_idx]
            intrinsics_files = self.all_intrinsics_files[real_idx]
            render_pose_files = self.render_pose_files[real_idx]
            # if self.mode == "train":
            # id_render = random.randint(0,len(rgb_files))
            # id_render = np.random.choice(np.arange(len(pose_files)))
            # elif self.mode == "test":
                # if len(id_render) > 10:
                #     id_render = id_render[:10]
            idx = idx % (len(render_rgb_files))
            id_render = idx
            img_idx = render_rgb_files[id_render][-8:-1]
            img_idx = img_idx.replace(".jp","") if ".jp" in img_idx else img_idx
            img_idx = img_idx.replace("r","") if "r" in img_idx else img_idx
            img_idx = img_idx.replace("o","") if "o" in img_idx else img_idx
            img_idx = img_idx.replace("l","") if "l" in img_idx else img_idx
            img_idx = img_idx.replace("/","") if "/" in img_idx else img_idx

            train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
            render_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in render_pose_files], axis=0)
            render_pose =  render_poses[id_render]
            # subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

            nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            self.num_source_views ,
            tar_id=id_render,
            angular_dist_method='dist',
        )
            # print(nearest_pose_ids)
            id_feat = np.random.choice(nearest_pose_ids, min(self.num_source_views, len(nearest_pose_ids)), replace=False)
            # id_feat_pool = get_nearest_pose_ids(
            #     render_pose,
            #     train_poses,
            #     self.num_source_views * subsample_factor,
            #     tar_id=id_render,
            #     angular_dist_method="vector",
            # )
            # id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

            if id_render in id_feat:
                assert id_render not in id_feat
            # occasionally include input image
            if np.random.choice([0, 1], p=[0.995, 0.005]) :
                id_feat[np.random.choice(len(id_feat))] = id_render

            rgb = imageio.imread(render_rgb_files[id_render]).astype(np.float32) / 255.0
                          
            img = Image.open(depth_files[id_render])
            depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
            depth = np.ascontiguousarray(depth, dtype=np.float32)
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

            intrinsics = np.loadtxt(intrinsics_files[id_render]).reshape([4, 4])

            img_size = rgb.shape[:2]
            camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
                np.float32
            )

            # img = Image.open(label_files[id_render])
            # label = np.asarray(img, dtype=np.int32)
            # label = np.ascontiguousarray(label)
            # label = cv2.resize(label, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            # label = label.astype(np.int32)
            # # label = self.scan2nyu[label]
            # label = self.label_mapping(label)

            all_poses = [render_pose]
            # get depth range
            # poses = render_pose[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
            # bds = render_pose[:, -2:].transpose([1, 0])
            # bds = np.moveaxis(bds, -1, 0).astype(np.float32)
            # far_depth = origin_depth + max_radius
            # depth_range = torch.tensor([near_depth, far_depth])
            depth_range = torch.tensor([0.1, 10.0])

            src_rgbs = []
            src_cameras = []
            src_intrinsics = []
            src_extrinsics = []
            # src_labels = []
            for id in id_feat:
                src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
                
                # src_label = Image.open(label_files[id])
                # src_label = np.asarray(src_label, dtype=np.int32)
                # src_label = np.ascontiguousarray(src_label)
                # src_label = cv2.resize(src_label, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
                # src_label = src_label.astype(np.int32)
                # src_label = self.scan2nyu[src_label]
                # src_label = self.label_mapping(src_label)

                pose = np.loadtxt(pose_files[id]).reshape(4, 4)

                if self.rectify_inplane_rotation:
                    pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

                src_rgbs.append(src_rgb)
                intrinsics = np.loadtxt(intrinsics_files[id]).reshape([4, 4])
                img_size = src_rgb.shape[:2]
                src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
                    np.float32
                )
                
                src_cameras.append(src_camera)
                src_extrinsics.append(pose)
                # src_labels.append(src_label)

            src_rgbs = np.stack(src_rgbs)
            src_cameras = np.stack(src_cameras)
            src_extrinsics = np.stack(src_extrinsics)

            #分界线
            rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics = loader_resize(rgb,camera.copy(),src_rgbs,src_cameras.copy(), size=self.image_size)
            src_extrinsics = torch.from_numpy(src_extrinsics).float()
            extrinsics = torch.from_numpy(render_pose).unsqueeze(0).float()
            
            src_intrinsics = self.normalize_intrinsics(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size)
            intrinsics = self.normalize_intrinsics(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size)

            depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

            # Resize the world to make the baseline 1.
            if src_extrinsics.shape[0] == 2:
                a, b = src_extrinsics[:, :3, 3]
                scale = (a - b).norm()
                if scale < 0.001:
                    print(
                        f"Skipped {scene} because of insufficient baseline "
                        f"{scale:.6f}"
                    )
                src_extrinsics[:, :3, 3] /= scale
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1

            return {    'rgb': torch.from_numpy(rgb[..., :3]),
                        'camera': torch.from_numpy(camera),
                        "rgb_path": render_rgb_files[id_render],
                        'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                        'src_cameras': torch.from_numpy(src_cameras),
                        'depth_range': depth_range,
                        'idx': idx,
                        'scaled_shape': (0, 0), # (378, 504)
                        "context": {
                                "extrinsics": src_extrinsics,
                                "intrinsics": src_intrinsics,
                                "image": torch.from_numpy(src_rgbs[..., :3]).permute(0, 3, 1, 2),
                                "near":  (depth_range[0].repeat(self.num_source_views) / scale).float(),
                                "far": (depth_range[1].repeat(self.num_source_views) / scale).float(),
                                "index": torch.from_numpy(id_feat),
                                # "labels": torch.tensor(src_labels)
                        },                                                                                        
                        "target": {
                                "extrinsics": extrinsics,
                                "intrinsics": intrinsics,
                                "image": torch.from_numpy(rgb[..., :3]).unsqueeze(0).permute(0, 3, 1, 2),
                                "near": (depth_range[0].unsqueeze(0) / scale).float(),
                                "far": (depth_range[1].unsqueeze(0) / scale).float(),
                                "index": torch.tensor([int(img_idx)]),
                                # "labels": torch.tensor(label)
                        
                        },"scene":scene}
        