"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np

import mindspore.common.dtype as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.config import common_config, KingsCollege, StMarysChurch
from src.posenet import PoseNet

parser = argparse.ArgumentParser(description='PoseNet')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument('--dataset', type=str, default='KingsCollege',
                    choices=['KingsCollege', 'StMarysChurch'],
                    help='Name of dataset.')
parser.add_argument("--ckpt_url", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="posenet", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["MINDIR", "AIR"], default='MINDIR', help='file format')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    cfg = common_config
    if args.dataset == 'KingsCollege':
        dataset_cfg = KingsCollege
    elif args.dataset == 'StMarysChurch':
        dataset_cfg = StMarysChurch

    net = PoseNet()

    assert cfg.checkpoint_dir is not None, "cfg.checkpoint_dir is None."
    param_dict = load_checkpoint(args.ckpt_url)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.ones([dataset_cfg.batch_size, 3, 224, 224]), ms.float32)
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
