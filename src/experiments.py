import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data import OfficeHomeDataset
from models import CLIPModel
from config import domains

class Experiments:
    def __init__(self, model_name: str, domain: str, frac: float = 1.0) -> None:
        """model_name is vitb16 or vitb32, domain is one Office-Home domain.
        frac is fraction of each split in 0 to 1.0, use it only for quick testing"""
        self.model_name = model_name
        self.domain = domain
        self.frac = frac

        self.cwd = Path.cwd()
        self.output_dir = Path(self.cwd, "outputs")
        self.feature_dir = Path(self.output_dir, "features")
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        # CLIP is loaded only when features have to be extracted
        self.model = None
        self.datasets = {s: OfficeHomeDataset(domain=self.domain, split=s, frac=self.frac) for s in ["dev", "test", "heldout"]}
        self.num_classes = len(self.datasets["test"].class_names)

    def feature_path(self, split: str) -> Path:
        """it returns where the features of a split are saved"""
        return Path(self.feature_dir, f"{self.model_name}_{self.domain.replace(' ', '_')}_frac{self.frac}_{split}.npz")

    def extract_features(self, split: str, batch_size: int = 128) -> None:
        """it runs CLIP on the original and the flipped image of every item in a split and saves the
        features and logits, so CLIP never has to run again for this split"""
        if self.model is None:
            self.model = CLIPModel(model_name=self.model_name)
        ds = self.datasets[split]
        text_feats = self.model.encode_text(ds.prompt_names())

        feats0, feats1 = [], []
        for start in tqdm(range(0, len(ds), batch_size), f"Extracting {self.domain} {split}", unit="batches"):
            images = [OfficeHomeDataset._load_image(ds[i]["image_path"]) for i in range(start, min(start + batch_size, len(ds)))]
            feats0.append(self.model.encode_images(images))
            feats1.append(self.model.encode_images([OfficeHomeDataset._flip_image(i) for i in images]))
        feats0, feats1 = torch.cat(feats0), torch.cat(feats1)

        features = {
            "feats0": feats0.cpu().numpy(),
            "feats1": feats1.cpu().numpy(),
            "logits0": CLIPModel.get_logits(feats0, text_feats).cpu().numpy(),
            "logits1": CLIPModel.get_logits(feats1, text_feats).cpu().numpy(),
            "labels": np.array([i["label"] for i in ds.data]),
            "image_ids": np.array([i["image_id"] for i in ds.data])
        }
        np.savez(self.feature_path(split), **features)
        print(f"Features saved at {self.feature_path(split)}")

    def load_features(self, split: str) -> dict:
        """it loads the saved features of a split, and extracts them first if they are not saved yet"""
        if not self.feature_path(split).exists():
            self.extract_features(split)
        with np.load(self.feature_path(split)) as f:
            features = {key: f[key] for key in f.files}

        return features

def main(model_name: str, domain: str, frac: float) -> None:
    exp = Experiments(model_name=model_name, domain=domain, frac=frac)
    for split in ["dev", "test", "heldout"]:
        features = exp.load_features(split)
        zero_shot = 100 * np.mean(features["logits0"].argmax(axis=1) == features["labels"])
        agree = 100 * np.mean(features["logits0"].argmax(axis=1) == features["logits1"].argmax(axis=1))
        print(f"{domain}, {split}: feats {features['feats0'].shape}, zero-shot {zero_shot:.2f}%, views agree {agree:.1f}%")

if __name__ == "__main__":
    for domain in domains:
        main("vitb16", domain, 1.0)