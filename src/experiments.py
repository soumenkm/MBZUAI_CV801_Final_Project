import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data import OfficeHomeDataset
from models import CLIPModel
from config import domains, k, alpha, beta, target_rates, gamma_grid, order_seeds, adaptive_lam, conditions
import json

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

        self.run_dir = Path(self.output_dir, "runs", f"{self.model_name}_{self.domain.replace(' ', '_')}_frac{self.frac}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

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

    def calibrate_offset(self, rate: float) -> float:
        """the adaptive rule has no direct threshold, so this finds the offset that makes it admit
        about the target share of dev images. then it can be compared with the other rules at the same rate"""
        signals = self.signals["dev"]
        start_mean = (signals["entropy"] / np.log(self.num_classes)).mean()
        best_offset, best_gap = 0.0, np.inf
        for offset in np.linspace(-0.5, 0.5, 101):
            state = {"class_means": np.full(self.num_classes, start_mean), "offset": offset, "lam": adaptive_lam}
            allowed = []
            for i in range(len(signals["pred"])):
                allowed.append(CLIPModel.is_allowed("adaptive", signals, i, state))
                CLIPModel.update_class_means(state, signals, i)
            if abs(np.mean(allowed) - rate) < best_gap:
                best_offset, best_gap = offset, abs(np.mean(allowed) - rate)

        return best_offset

    def make_state(self, rule: str, rate: float, seed: int) -> dict:
        """it sets up what an admission rule needs before a run. thresholds come from the dev split,
        so that the rule admits about the target share of images. nothing from the test split is used"""
        dev = self.signals["dev"]
        if rule in ["js", "mean_entropy", "entropy_of_mean"]:
            return {"threshold": np.quantile(dev[rule], rate)}
        if rule == "random":
            return {"rng": np.random.default_rng(seed), "rate": rate}
        if rule == "adaptive":
            start_mean = (dev["entropy"] / np.log(self.num_classes)).mean()
            return {"class_means": np.full(self.num_classes, start_mean), "offset": self.calibrate_offset(rate), "lam": adaptive_lam}

        return {}

    def run_path(self, name: str, rate: float, gamma: float, order_seed: int) -> Path:
        """it returns where the result of one run is saved"""
        return Path(self.run_dir, f"{name}_rate{rate}_gamma{gamma}_order{order_seed}.json")

    def run_condition(self, name: str, rate: float, gamma: float, order_seed: int) -> dict:
        """it runs one condition (for example P1) at one admission rate, gamma and stream order,
        saves the full result and returns a short summary"""
        rule = conditions[name]["rule"]
        # B0 is zero-shot CLIP, the cache is still filled but it has no effect on the prediction
        alpha_used = 0.0 if name == "B0" else alpha
        state = self.make_state(rule, rate, seed=order_seed)
        rows, snapshots = self.run_stream(rule, state, gamma, alpha_used, order_seed)

        result = {"name": name, "rule": rule, "rate": rate, "gamma": gamma, "order_seed": order_seed,
                  "k": k, "alpha": alpha_used, "beta": beta, "rows": rows, "snapshots": snapshots}
        path = self.run_path(name, rate, gamma, order_seed)
        temp_path = path.with_suffix(".temp")
        with open(temp_path, mode="w", encoding="utf-8") as f:
            json.dump(result, f)
        temp_path.replace(path)

        summary = {"name": name, "rate": rate, "gamma": gamma, "order_seed": order_seed,
                   "accuracy": Experiments.get_accuracy(rows), "admitted": 100 * np.mean([r["allowed"] for r in rows])}

        return summary

    def run_all(self) -> list:
        """it runs every condition at every admission rate, gamma and stream order"""
        rate_rules = ["js", "mean_entropy", "entropy_of_mean", "random", "adaptive"]
        runs = []
        for name, cond in conditions.items():
            rates = target_rates if cond["rule"] in rate_rules else [None]
            gammas = gamma_grid if cond["use_gamma"] else [0.0]
            for rate in rates:
                for gamma in gammas:
                    for order_seed in order_seeds:
                        runs.append((name, rate, gamma, order_seed))

        summaries = [self.run_condition(*run) for run in tqdm(runs, f"Running {self.domain}", unit="runs")]

        return summaries

    def check_invariants(self) -> None:
        """it checks things that must hold if the code is right: B0 equals zero-shot CLIP, and the readout
        switch never changes what is stored, so C0 and P2 store the same images, and so do P1 and P3"""
        def load(name, rate, gamma, order_seed):
            with open(self.run_path(name, rate, gamma, order_seed), mode="r", encoding="utf-8") as f:
                return json.load(f)

        zero_shot = 100 * np.mean(self.signals["test"]["pred"] == self.features["test"]["labels"])
        b0 = Experiments.get_accuracy(load("B0", None, 0.0, order_seeds[0])["rows"])
        print(f"B0: {b0:.2f}%, zero-shot: {zero_shot:.2f}%, same: {np.isclose(b0, zero_shot)}")

        gamma = gamma_grid[-1]
        for order_seed in order_seeds:
            c0_vs_p2 = load("C0", None, 0.0, order_seed)["snapshots"] == load("P2", None, gamma, order_seed)["snapshots"]
            p1_vs_p3 = all(load("P1", r, 0.0, order_seed)["snapshots"] == load("P3", r, gamma, order_seed)["snapshots"] for r in target_rates)
            print(f"order {order_seed}: C0 and P2 store the same: {c0_vs_p2}, P1 and P3 store the same: {p1_vs_p3}")

def main(model_name: str, domain: str, frac: float) -> None:
    exp = Experiments(model_name=model_name, domain=domain, frac=frac)
    # for split in ["dev", "test", "heldout"]:
    #     features = exp.load_features(split)
    #     zero_shot = 100 * np.mean(features["logits0"].argmax(axis=1) == features["labels"])
    #     agree = 100 * np.mean(features["logits0"].argmax(axis=1) == features["logits1"].argmax(axis=1))
    #     print(f"{domain}, {split}: feats {features['feats0'].shape}, zero-shot {zero_shot:.2f}%, views agree {agree:.1f}%")

    # with alpha 0 the cache has no effect, so it must give exactly the zero-shot accuracy
    # zero_shot = 100 * np.mean(exp.signals["test"]["pred"] == exp.features["test"]["labels"])
    # for order_seed in [0, 1, 2]:
    #     b0_rows, _ = exp.run_stream("always", {}, gamma=0.0, alpha_used=0.0, order_seed=order_seed)
    #     c0_rows, _ = exp.run_stream("always", {}, gamma=0.0, alpha_used=alpha, order_seed=order_seed)
    #     print(f"{domain}, order {order_seed}: B0 {Experiments.get_accuracy(b0_rows):.2f}% (zero-shot {zero_shot:.2f}%), C0 {Experiments.get_accuracy(c0_rows):.2f}%, cache size at end: {c0_rows[-1]['cache_size']} of {k * exp.num_classes}")

    summaries = exp.run_all()
    exp.check_invariants()

    # # short table at the 50% admission rate and gamma 16, averaged over the stream orders
    # for name in conditions:
    #     picked = [s for s in summaries if s["name"] == name and s["rate"] in [None, 0.5] and s["gamma"] in [0.0, 16]]
    #     print(f"{domain}, {name}: accuracy {np.mean([s['accuracy'] for s in picked]):.2f}%, admitted {np.mean([s['admitted'] for s in picked]):.1f}%")

if __name__ == "__main__":
    for domain in domains:
        main("vitb16", domain, 1.0)