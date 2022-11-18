"""
preprocess script
"""
import os
import argparse
from src.dataset import create_posenet_dataset
from src.config import KingsCollege

parser = argparse.ArgumentParser(description='PoseNet')
parser.add_argument("--result_path", type=str, default="./preprocess_Result1/", help="result path")
args = parser.parse_args()

def preprocess(result_path):
    dataset = create_posenet_dataset(KingsCollege.mindrecord_dir, 1, 1, True, 0)

    img_path = os.path.join(result_path, "img_data")
    label_path = os.path.join(result_path, "label")
    os.makedirs(img_path)
    os.makedirs(label_path)

    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        lable_data = data["label"]

        file_name = "posenet" + "_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        lable_file_path = os.path.join(lable_path, file_name)
        lable_data.tofile(lable_file_path)

    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    preprocess(args.result_path)
