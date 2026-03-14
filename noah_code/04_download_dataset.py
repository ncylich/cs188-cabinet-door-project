"""
Step 4: Download the OpenCabinet Dataset
==========================================
Downloads pre-collected demonstration data for the OpenCabinet task.
This includes human demonstrations and MimicGen-expanded trajectories
across diverse kitchen scenes.

Usage:
    python 04_download_dataset.py

The dataset will be downloaded to the default RoboCasa datasets directory.
"""

import os
import robocasa  # noqa: F401
from robocasa.scripts.download_datasets import download_datasets
from robocasa.utils.dataset_registry_utils import get_ds_path


def main():
    print("=" * 60)
    print("  OpenCabinet - Dataset Download")
    print("=" * 60)

    task = "OpenCabinet"

    # Show what datasets are available
    print(f"\nDataset paths for {task}:")
    for source in ["human", "mg"]:
        path = get_ds_path(task, source=source)
        if path is not None:
            exists = os.path.exists(path)
            status = "EXISTS" if exists else "NOT DOWNLOADED"
            print(f"  {source:8s}: {path}")
            print(f"           [{status}]")
        else:
            print(f"  {source:8s}: not registered")

    # Download the dataset
    print(f"\nDownloading {task} datasets...")
    print("(This may take a while depending on your connection)\n")

    try:
        download_datasets(
            tasks=[task],
            split=["pretrain"],
            source=["human"],
        )
        print(f"\nHuman demonstration dataset downloaded successfully!")
    except Exception as e:
        print(f"Download failed: {e}")
        print("You may need to check your network connection.")
        return

    # Verify the download
    human_path = get_ds_path(task, source="human")
    if human_path and os.path.exists(human_path):
        print(f"\nDataset location: {human_path}")

        # Show dataset contents
        print("\nDataset structure:")
        for root, dirs, files in os.walk(human_path):
            level = root.replace(human_path, "").count(os.sep)
            indent = "  " * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            if level < 2:
                for f in files[:5]:
                    print(f"{indent}  {f}")
                if len(files) > 5:
                    print(f"{indent}  ... ({len(files)} files total)")
    else:
        print("Warning: dataset path not found after download.")

    print("\nDone! Proceed to 05_playback_demonstrations.py")


if __name__ == "__main__":
    main()
