"""Download the preprocessed S3DIS dataset and the official PTv3 S3DIS checkpoint (Hugging Face Hub).

    python scripts/download.py --data data --weights weights

S3DIS comes from Pointcept/s3dis-compressed (2 GB archive, 8 GB unpacked). By downloading it you
accept the S3DIS terms of use (http://buildingparser.stanford.edu/dataset.html). The checkpoint
is Pointcept/PointTransformerV3: s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth (555 MB).
"""
import argparse
import os
import tarfile

from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()

    path = hf_hub_download("Pointcept/PointTransformerV3", "s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth",
                           local_dir=args.weights)
    print("checkpoint:", path)
    if args.skip_data:
        return
    archive = hf_hub_download("Pointcept/s3dis-compressed", "s3dis.tar.gz", repo_type="dataset", local_dir=args.data)
    target = os.path.join(args.data, "s3dis")
    os.makedirs(target, exist_ok=True)
    with tarfile.open(archive) as tar:
        root = os.path.realpath(target)
        for member in tar.getmembers():                      # refuse paths that escape the target folder
            if not os.path.realpath(os.path.join(target, member.name)).startswith(root + os.sep):
                raise ValueError(f"unsafe path in archive: {member.name}")
        tar.extractall(target)
    os.remove(archive)
    print("S3DIS:", target, sorted(os.listdir(target)))


if __name__ == "__main__":
    main()
