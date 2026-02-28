"""Download HuggingFace datasets to disk for use with preprocess_data_packing.py.

Available datasets (pass one or more to --datasets):
  nemotron-math     nvidia/Nemotron-CC-Math-v1 (4plus quality filter)
  sft-general       nvidia/Nemotron-Pretraining-SFT-v1 (general)
  sft-code          nvidia/Nemotron-Pretraining-SFT-v1 (code)
  sft-math          nvidia/Nemotron-Pretraining-SFT-v1 (math)
  fineweb-edu       HuggingFaceFW/fineweb-edu (sample-350BT)

Usage:
    python utils/download_ds.py --dataset-path /path/to/raw_data --datasets nemotron-math
    python utils/download_ds.py --dataset-path /path/to/raw_data --datasets nemotron-math sft-general
"""

import argparse
from pathlib import Path

from datasets import load_dataset

AVAILABLE = ["nemotron-math", "sft-general", "sft-code", "sft-math", "fineweb-edu"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, required=True, help="Root directory for raw dataset storage.")
parser.add_argument(
    "--datasets",
    nargs="+",
    choices=AVAILABLE,
    required=True,
    metavar="DATASET",
    help=f"Which dataset(s) to download. Choices: {AVAILABLE}",
)
args = parser.parse_args()

dataset_path = Path(args.dataset_path)
dataset_path.mkdir(parents=True, exist_ok=True)

for name in args.datasets:
    if name == "nemotron-math":
        print("Downloading nvidia/Nemotron-CC-Math-v1 (4plus) ...")
        ds = load_dataset("nvidia/Nemotron-CC-Math-v1", "4plus")
        ds.save_to_disk(str(dataset_path / "Nemotron-CC-Math-v1-4plus"))
        print(f"  Saved to {dataset_path / 'Nemotron-CC-Math-v1-4plus'}")

    elif name == "sft-general":
        print("Downloading nvidia/Nemotron-Pretraining-SFT-v1 (General) ...")
        ds = load_dataset("nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-General")
        ds.save_to_disk(str(dataset_path / "Nemotron-Pretraining-SFT-v1-General"))
        print(f"  Saved to {dataset_path / 'Nemotron-Pretraining-SFT-v1-General'}")

    elif name == "sft-code":
        print("Downloading nvidia/Nemotron-Pretraining-SFT-v1 (Code) ...")
        ds = load_dataset("nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-Code")
        ds.save_to_disk(str(dataset_path / "Nemotron-Pretraining-SFT-v1-Code"))
        print(f"  Saved to {dataset_path / 'Nemotron-Pretraining-SFT-v1-Code'}")

    elif name == "sft-math":
        print("Downloading nvidia/Nemotron-Pretraining-SFT-v1 (Math) ...")
        ds = load_dataset("nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-MATH")
        ds.save_to_disk(str(dataset_path / "Nemotron-Pretraining-SFT-v1-Math"))
        print(f"  Saved to {dataset_path / 'Nemotron-Pretraining-SFT-v1-Math'}")

    elif name == "fineweb-edu":
        print("Downloading HuggingFaceFW/fineweb-edu (sample-350BT) ...")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", split="train")
        ds.save_to_disk(str(dataset_path / "fineweb-edu-350BT"))
        print(f"  Saved to {dataset_path / 'fineweb-edu-350BT'}")

print("Done.")
