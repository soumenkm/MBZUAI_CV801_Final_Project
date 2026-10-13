from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from config import domains, split_fracs

import random
import math
from PIL import Image, ImageOps

class OfficeHomeDataset(Dataset):
    def __init__(self, domain: str, split: str, frac: float = 1.0, seed: int = 42) -> None:
        """domain is one of Art, Clipart, Product, Real World
        split is either all, dev, test or heldout
        frac is fraction of the split in 0 to 1.0, use it only for quick testing
        seed fixes the split so that it is the same in every run"""
        if domain not in domains:
            raise ValueError(f"Unknown domain: {domain}!")
        if split not in ["all", "dev", "test", "heldout"]:
            raise ValueError(f"Unknown split: {split}!")

        self.domain = domain
        self.split = split
        self.frac = frac
        self.seed = seed

        self.cwd = Path.cwd()
        self.data_path = Path(self.cwd, "data/OfficeHomeDataset")
        self.domain_path = Path(self.data_path, self.domain)

        self.class_names = self._load_class_names()
        self.data = self._create_data()

    def _load_class_names(self) -> list:
        """it reads the class names from the folder names. they are sorted so that
        the label ids are the same for every domain"""
        return sorted(i.name for i in self.domain_path.iterdir() if i.is_dir())

    def _load_items(self) -> list:
        """it lists all the images of the domain with their labels.
        it doesn't open the images here, they are opened on demand."""
        items = []
        for label, class_name in enumerate(self.class_names):
            for i in sorted(Path(self.domain_path, class_name).iterdir()):
                if i.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                item = {
                    "image_id": f"{self.domain}/{class_name}/{i.name}",
                    "image_path": str(i),
                    "label": label,
                    "class_name": class_name,
                    "domain": self.domain
                }
                items.append(item)

        return items

    def _create_data(self) -> list:
        """it splits the domain into dev, stream and probe sets class by class,
        so every split has every class in roughly the same proportion.
        then it keeps only frac of the split for quick testing"""
        items = self._load_items()
        if self.split == "all":
            data = items
        else:
            data = []
            for label in range(len(self.class_names)):
                class_items = [i for i in items if i["label"] == label]
                random.Random(self.seed + label).shuffle(class_items)
                n_dev = math.floor(len(class_items) * split_fracs["dev"])
                n_test = math.floor(len(class_items) * split_fracs["test"])
                # whatever is left after dev and test goes to heldout
                splits = {
                    "dev": class_items[:n_dev],
                    "test": class_items[n_dev:n_dev + n_test],
                    "heldout": class_items[n_dev + n_test:]
                }
                data.extend(splits[self.split])

        # Reduce the size of the data for testing
        random.Random(self.seed).shuffle(data)
        idx = math.floor(len(data) * self.frac)

        # the stream order is set by get_order(), not by the order of this list
        return data[0:idx]

    def get_order(self, order_seed: int) -> list:
        """it returns the item indices in a shuffled order. the stream is fed to the
        cache in this order, so a different order seed gives a different stream of the same images"""
        order = list(range(len(self.data)))
        random.Random(order_seed).shuffle(order)

        return order

    def prompt_names(self) -> list:
        """it returns the class names used in the text prompt, for example Alarm_Clock becomes alarm clock"""
        return [c.replace("_", " ").lower() for c in self.class_names]

    @staticmethod
    def _flip_image(image: Image) -> Image:
        """it mirrors the image left to right. this is the second view used to measure disagreement.
        CLIP's center crop is symmetric, so flipping before or after the preprocessing gives the same result."""
        return ImageOps.mirror(image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self):
            raise IndexError("Index is out of range!")

        return self.data[idx]

    @staticmethod
    def _load_image(image_path: str) -> Image:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        return image

def main(domain: str) -> None:
    # ds = OfficeHomeDataset(domain=domain)
    # image = OfficeHomeDataset._load_image(ds[0]["image_path"])
    # print(f"{domain}: {len(ds)} images, {len(ds.class_names)} classes")
    # print(ds[0], image.size)

    ds_all = OfficeHomeDataset(domain=domain, split="all")
    split_ids = {s: {i["image_id"] for i in OfficeHomeDataset(domain=domain, split=s).data} for s in ["dev", "test", "heldout"]}
    all_ids = {i["image_id"] for i in ds_all.data}
    overlap = (split_ids["dev"] & split_ids["test"]) | (split_ids["dev"] & split_ids["heldout"]) | (split_ids["test"] & split_ids["heldout"])
    missing = all_ids - (split_ids["dev"] | split_ids["test"] | split_ids["heldout"])

    sizes = ", ".join(f"{s}: {len(v)}" for s, v in split_ids.items())
    print(f"{domain}: {len(ds_all)} images, {len(ds_all.class_names)} classes, {sizes}, overlap: {len(overlap)}, missing: {len(missing)}")

    ds = OfficeHomeDataset(domain=domain, split="test")
    print(f"order 0: {ds.get_order(order_seed=0)[:5]}, order 1: {ds.get_order(order_seed=1)[:5]}")

    ds = OfficeHomeDataset(domain=domain, split="test")
    image = OfficeHomeDataset._load_image(ds[0]["image_path"])
    flipped = OfficeHomeDataset._flip_image(image)
    # flipping twice should give back the same image
    same = list(OfficeHomeDataset._flip_image(flipped).getdata()) == list(image.getdata())
    print(f"{domain}: {ds.prompt_names()[:3]}, flip size: {flipped.size}, flip twice same: {same}")

    ds_small = OfficeHomeDataset(domain=domain, split="test", frac=0.1)
    print(f"{domain}: test with frac=0.1 has {len(ds_small)} of {len(ds)} images")

if __name__ == "__main__":
    for domain in domains:
        main(domain)