from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from config import domains

class OfficeHomeDataset(Dataset):
    def __init__(self, domain: str) -> None:
        """domain is one of Art, Clipart, Product, Real World"""
        if domain not in domains:
            raise ValueError(f"Unknown domain: {domain}!")

        self.domain = domain

        self.cwd = Path.cwd()
        self.data_path = Path(self.cwd, "data/OfficeHomeDataset")
        self.domain_path = Path(self.data_path, self.domain)

        self.class_names = self._load_class_names()
        self.data = self._load_items()

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
    ds = OfficeHomeDataset(domain=domain)
    image = OfficeHomeDataset._load_image(ds[0]["image_path"])
    print(f"{domain}: {len(ds)} images, {len(ds.class_names)} classes")
    print(ds[0], image.size)

if __name__ == "__main__":
    for domain in domains:
        main(domain)