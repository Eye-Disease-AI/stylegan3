from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from dataset.loader import NuclearCataractDataset
import os
from torchvision.io import write_png
from tqdm import tqdm


def main():
    dest_dir = "nuclear_cataract_sg3"

    try:
        os.makedirs(dest_dir)
    except Exception:
        print("nuclear_cataract_sg3 already exists, exiting")
        return

    ncd = NuclearCataractDataset(
        NuclearCataractDataset.TrainValMode(1.0, 0.0)
    )

    class_mapping = {
        "labels": []
    }

    train_set = ncd.train_set()

    with ThreadPoolExecutor(max_workers=8) as executor, tqdm(total=len(train_set)*2) as pbar:
        futures = []

        for idx, (img, label) in enumerate(iter(train_set)):
            dest_path_rel = f"{idx}.png"
            futures.append(
                executor.submit(write_png, img, f"{dest_dir}/{dest_path_rel}")
            )
            class_mapping["labels"].append([
                dest_path_rel, label
            ])
            pbar.update(1)

        for _ in as_completed(futures):
            pbar.update(1)

    with open(f"{dest_dir}/dataset.json", "w") as dataset_json:
        json.dump(class_mapping, dataset_json)


if __name__ == "__main__":
    main()
