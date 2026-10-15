import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data import OfficeHomeDataset
from models import CLIPModel
from config import domains, k, alpha, beta

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

        self.features = {s: self.load_features(s) for s in ["dev", "test"]}
        self.signals = {s: self.get_signals(s) for s in ["dev", "test"]}

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

    def get_signals(self, split: str) -> dict:
        """it computes the prediction, entropy, JS and the other signals for every image of a split"""
        p0 = CLIPModel.get_probs(torch.from_numpy(self.features[split]["logits0"]))
        p1 = CLIPModel.get_probs(torch.from_numpy(self.features[split]["logits1"]))

        return CLIPModel.get_signals(p0, p1)

    def run_stream(self, rule: str, state: dict, gamma: float, alpha_used: float, order_seed: int) -> tuple:
        """it runs one pass over the test split in the given order. for each image it first predicts with
        the cache built from the earlier images, and only then decides whether the image goes into the cache.
        the true label is saved for the analysis but it is never used to make a decision here"""
        feats, signals = self.features["test"], self.signals["test"]
        cache = CLIPModel.new_cache(num_classes=self.num_classes, k=k, dim=feats["feats0"].shape[1])
        order = self.datasets["test"].get_order(order_seed)
        checkpoints = [len(order) // 4, len(order) // 2, len(order)]

        rows, snapshots = [], {}
        for t, i in enumerate(order):
            votes = CLIPModel.read_cache(cache, feats["feats0"][i], beta=beta, gamma=gamma)
            pred_final = int((feats["logits0"][i] + alpha_used * votes).argmax())

            # the cache always stores the plain CLIP prediction, not the final one, so that
            # changing the readout never changes what gets stored
            allowed = CLIPModel.is_allowed(rule, signals, i, state)
            action = "not allowed"
            if allowed:
                action = CLIPModel.add_to_cache(cache, feats["feats0"][i], int(signals["pred"][i]), signals["entropy"][i], signals["js"][i], i)
            if rule == "adaptive":
                CLIPModel.update_class_means(state, signals, i)

            rows.append({
                "t": t,
                "idx": int(i),
                "label": int(feats["labels"][i]),
                "pred_base": int(signals["pred"][i]),
                "pred_final": pred_final,
                "entropy": float(signals["entropy"][i]),
                "js": float(signals["js"][i]),
                "allowed": bool(allowed),
                "action": action,
                "cache_size": int((cache["labels"] >= 0).sum())
            })
            # which images are in the cache at 25%, 50% and 100% of the stream
            if t + 1 in checkpoints:
                snapshots[str(t + 1)] = cache["idx"][cache["labels"] >= 0].tolist()

        return rows, snapshots

    @staticmethod
    def get_accuracy(rows: list) -> float:
        """it returns the share of images whose final prediction is correct, in percent"""
        return 100 * np.mean([r["pred_final"] == r["label"] for r in rows])

def main(model_name: str, domain: str, frac: float) -> None:
    exp = Experiments(model_name=model_name, domain=domain, frac=frac)
    # for split in ["dev", "test", "heldout"]:
    #     features = exp.load_features(split)
    #     zero_shot = 100 * np.mean(features["logits0"].argmax(axis=1) == features["labels"])
    #     agree = 100 * np.mean(features["logits0"].argmax(axis=1) == features["logits1"].argmax(axis=1))
    #     print(f"{domain}, {split}: feats {features['feats0'].shape}, zero-shot {zero_shot:.2f}%, views agree {agree:.1f}%")

    # with alpha 0 the cache has no effect, so it must give exactly the zero-shot accuracy
    zero_shot = 100 * np.mean(exp.signals["test"]["pred"] == exp.features["test"]["labels"])
    for order_seed in [0, 1, 2]:
        b0_rows, _ = exp.run_stream("always", {}, gamma=0.0, alpha_used=0.0, order_seed=order_seed)
        c0_rows, _ = exp.run_stream("always", {}, gamma=0.0, alpha_used=alpha, order_seed=order_seed)
        print(f"{domain}, order {order_seed}: B0 {Experiments.get_accuracy(b0_rows):.2f}% (zero-shot {zero_shot:.2f}%), C0 {Experiments.get_accuracy(c0_rows):.2f}%, cache size at end: {c0_rows[-1]['cache_size']} of {k * exp.num_classes}")

if __name__ == "__main__":
    for domain in domains:
        main("vitb16", domain, 1.0)