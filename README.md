export CUDA_VISIBLE_DEVICES=6
python train_dgaussian.py ++rootdir=data/ibrnet/train \
    +ckpt_path=model_zoo/dgaussian_resnet.pth ++eval_scenes=[room] \
    ++train_scenes=[room] ++num_source_views=8 \
    ++expname=dgaussian_w_depth_room ++use_depth_loss=True


export CUDA_VISIBLE_DEVICES=4
python train_dgaussian.py \
    +ckpt_path=model_zoo/dgaussian_resnet.pth ++eval_scenes=[room] \
    ++num_source_views=5 \
    ++expname=pretrain_dgaussian ++use_depth_loss=False ++use_pred_pose=False

export CUDA_VISIBLE_DEVICES=0
python train_dgaussian.py \
    +ckpt_path=model_zoo/dgaussian_resnet.pth ++eval_scenes=[room] \
    ++num_source_views=5 \
    ++expname=pretrain_dgaussian_joint_pred ++use_depth_loss=False ++use_pred_pose=True



export CUDA_VISIBLE_DEVICES=3
python finetune_dgaussian_stable.py ++expname=finetune_dgaussian_room_depth +ckpt_path=data/ibrnet/train/out/finetune_dgaussian_stable_room/model/model_005000.pth ++use_depth_loss=True 

export CUDA_VISIBLE_DEVICES=3
python finetune_dgaussian_stable.py ++expname=finetune_dgaussian_room_depth_pose +ckpt_path=data/ibrnet/train/out/finetune_dgaussian_stable_room/model/model_005000.pth ++use_depth_loss=True ++use_pred_pose=True


export CUDA_VISIBLE_DEVICES=2
python train_dbarf.py --config configs/pretrain_dbarf.txt --rootdir data/ibrnet/train --ckpt_path model_zoo/dbarf_model_200000.pth --expname pretrain_dbarf_504 --num_source_views 5