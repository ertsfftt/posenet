"""
network config setting
"""
from easydict import EasyDict as edict

common_config = edict({
    'device_id': 0,
    'pre_trained': True,
    'max_steps': 30000,
    'save_checkpoint': True,
    # 'pre_trained_file': '/home/work/user-job-dir/posenet/pre_trained_googlenet_imagenet.ckpt',
    'pre_trained_file': '../pre_trained_googlenet_imagenet.ckpt',
    'checkpoint_dir': '../checkpoint',
    'save_checkpoint_epochs': 5,
    'keep_checkpoint_max': 10
})

KingsCollege = edict({
    'batch_size': 75,
    'lr_init': 0.001,
    'weight_decay': 0.5,
    'name': 'KingsCollege',
    'dataset_path': '../KingsCollege/',
    'mindrecord_dir': '../MindrecordKingsCollege'
})

StMarysChurch = edict({
    'batch_size': 75,
    'lr_init': 0.001,
    'weight_decay': 0.5,
    'name': 'StMarysChurch',
    'dataset_path': '../StMarysChurch/',
    'mindrecord_dir': '../MindrecordStMarysChurch'
})
