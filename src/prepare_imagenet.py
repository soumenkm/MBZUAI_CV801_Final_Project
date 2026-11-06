import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class ImageNetPrep:
    def __init__(self) -> None:
        """it puts ImageNet-R and ImageNet-A into the same layout as Office-Home, <root>/<domain>/<class>/<image>,
        so the same dataset class can read all three datasets"""
        self.cwd = Path.cwd()
        self.r_parquet_dir = Path(self.cwd, "data", "imagenet-r-hf")
        self.r_out_dir = Path(self.cwd, "data", "ImageNetR", "ImageNet-R")
        self.a_src_dir = Path(self.cwd, "data", "imagenet-a")
        self.a_out_dir = Path(self.cwd, "data", "ImageNetA", "ImageNet-A")

    def save_imagenet_r(self) -> int:
        """it writes every ImageNet-R image into a folder named after its class and returns the number of images"""
        n = 0
        for path in tqdm(sorted(self.r_parquet_dir.rglob("*.parquet")), "Saving ImageNet-R", unit="files"):
            df = pd.read_parquet(path)
            for row in df.itertuples():
                class_dir = Path(self.r_out_dir, row.class_name)
                class_dir.mkdir(parents=True, exist_ok=True)
                # the image column holds the original file bytes, so they are written without decoding
                Path(class_dir, f"{row.wnid}_{n:05d}.jpg").write_bytes(row.image["bytes"])
                n += 1

        return n

    def load_a_names(self) -> dict:
        """it reads the WordNet id and class name of each ImageNet-A class from the README of the release"""
        names = {}
        for line in Path(self.a_src_dir, "README.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("n") and parts[0][1:].isdigit():
                names[parts[0]] = "_".join(parts[1:])

        return names

    def save_imagenet_a(self) -> int:
        """it copies each ImageNet-A class folder, which is named by WordNet id, to a folder named after the class"""
        names = self.load_a_names()
        n = 0
        for wnid, name in tqdm(names.items(), "Saving ImageNet-A", unit="classes"):
            shutil.copytree(Path(self.a_src_dir, wnid), Path(self.a_out_dir, name), dirs_exist_ok=True)
            n += len(list(Path(self.a_out_dir, name).iterdir()))

        return n

def main() -> None:
    prep = ImageNetPrep()
    n = prep.save_imagenet_r()
    print(f"ImageNet-R: saved {n} images in {len(list(prep.r_out_dir.iterdir()))} classes (expected 30000 and 200)")
    n = prep.save_imagenet_a()
    print(f"ImageNet-A: saved {n} images in {len(list(prep.a_out_dir.iterdir()))} classes (expected 7500 and 200)")

if __name__ == "__main__":
    main()