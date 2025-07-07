import argparse
import os
import shutil
import sys
from glob import glob

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mimic_cxr_jpg_path",
    default="datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
)
parser.add_argument(
    "--save_dir",
    default="datasets/MIMIC-CXR/images",
)


def main(args):
    args = parser.parse_args(args)

    os.makedirs(args.save_dir, exist_ok=True)

    image_files = glob(os.path.join(args.mimic_cxr_jpg_path, "*", "*", "*", "*.jpg"))
    for image_file in tqdm(image_files):
        shutil.copy2(image_file, args.save_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
