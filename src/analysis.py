import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from models import CLIPModel
from config import domains

class Analysis:
    def __init__(self, model_name: str, frac: float = 1.0) -> None:
        """model_name and frac pick which saved features and runs are analysed"""
        self.model_name = model_name
        self.frac = frac

        self.cwd = Path.cwd()
        self.output_dir = Path(self.cwd, "outputs")
        self.feature_dir = Path(self.output_dir, "features")
        self.table_dir = Path(self.output_dir, "tables")
        self.fig_dir = Path(self.output_dir, "figures")
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.signals = {d: self.load_signals(d) for d in domains}

    def domain_tag(self, domain: str) -> str:
        """it returns the name part shared by the feature and run files of a domain"""
        return f"{self.model_name}_{domain.replace(' ', '_')}_frac{self.frac}"

    def load_signals(self, domain: str) -> dict:
        """it loads the saved test features of a domain, computes the signals and marks
        whether the zero-shot prediction is correct"""
        with np.load(Path(self.feature_dir, f"{self.domain_tag(domain)}_test.npz")) as f:
            logits0, logits1, labels = f["logits0"], f["logits1"], f["labels"]
        p0 = CLIPModel.get_probs(torch.from_numpy(logits0))
        p1 = CLIPModel.get_probs(torch.from_numpy(logits1))
        signals = CLIPModel.get_signals(p0, p1)
        signals["correct"] = signals["pred"] == labels

        return signals

    @staticmethod
    def entropy_bins(entropy: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """it puts each image into an entropy decile, 0 is the most confident tenth of the images"""
        edges = np.quantile(entropy, np.linspace(0, 1, n_bins + 1))

        return np.clip(np.searchsorted(edges, entropy, side="right") - 1, 0, n_bins - 1)

    @staticmethod
    def within_bin_auroc(signals: dict, n_bins: int = 10) -> float:
        """it checks if JS still separates correct from wrong predictions among images with similar entropy.
        it returns the mean AUROC of JS over the entropy deciles, 0.5 means JS adds nothing beyond entropy"""
        bins = Analysis.entropy_bins(signals["entropy"], n_bins)
        aurocs = []
        for b in range(n_bins):
            correct = signals["correct"][bins == b]
            # AUROC is not defined if a bin has only correct or only wrong predictions
            if 0 < correct.sum() < len(correct):
                aurocs.append(roc_auc_score(correct, -signals["js"][bins == b]))

        return float(np.mean(aurocs))

    @staticmethod
    def bootstrap_within_bin(signals: dict, n: int = 500) -> tuple:
        """it resamples the images with replacement and returns the 2.5% and 97.5% values of the within-bin AUROC"""
        rng = np.random.default_rng(42)
        values = []
        for _ in range(n):
            idx = rng.integers(0, len(signals["js"]), size=len(signals["js"]))
            values.append(Analysis.within_bin_auroc({key: v[idx] for key, v in signals.items()}))

        return np.quantile(values, 0.025), np.quantile(values, 0.975)

    def signal_table(self) -> pd.DataFrame:
        """RQ1 table: how much the two views differ in each domain, and how well each score separates
        correct from wrong zero-shot predictions (AUROC, 0.5 is chance level)"""
        rows = []
        for domain in domains:
            s = self.signals[domain]
            row = {
                "domain": domain,
                "zero_shot_acc": 100 * s["correct"].mean(),
                "mean_js": s["js"].mean(),
                "js_below_0.01": 100 * (s["js"] < 0.01).mean(),
                "views_agree": 100 * s["agree"].mean()
            }
            # low entropy or low JS should mean a correct prediction, hence the minus sign
            for score in ["entropy", "mean_entropy", "entropy_of_mean", "js"]:
                row[f"auroc_{score}"] = roc_auc_score(s["correct"], -s[score])
            row["auroc_agree"] = roc_auc_score(s["correct"], s["agree"])
            row["auroc_js_within_entropy_bins"] = Analysis.within_bin_auroc(s)
            row["auroc_js_within_low"], row["auroc_js_within_high"] = Analysis.bootstrap_within_bin(s)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.loc[len(df)] = ["Mean"] + df.iloc[:, 1:].mean().tolist()

        path = Path(self.table_dir, f"signals_{self.model_name}.csv")
        df.round(4).to_csv(path, index=False)
        print(f"Table saved at {path}")

        return df

    def plot_js_hist(self) -> None:
        """it plots the distribution of JS between the two views for each domain, on a log scale"""
        fig, axs = plt.subplots(1, len(domains), figsize=(4 * len(domains), 3.5), squeeze=False)
        for ax, domain in zip(axs[0], domains):
            js = self.signals[domain]["js"]
            ax.hist(np.log10(np.clip(js, 1e-6, None)), bins=50)
            ax.set_title(f"{domain} (N={len(js)})")
            ax.set_xlabel("log10 JS between original and flipped view")
            ax.set_ylabel("Number of images")
        plt.tight_layout()

        path = Path(self.fig_dir, f"js_hist_{self.model_name}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Figure saved at {path}")

    def plot_within_bin(self) -> None:
        """it plots zero-shot accuracy in each entropy decile, separately for images with JS below and
        above the median JS of that decile. a gap between the two lines means JS adds information"""
        fig, axs = plt.subplots(1, len(domains), figsize=(4 * len(domains), 3.5), squeeze=False)
        for ax, domain in zip(axs[0], domains):
            s = self.signals[domain]
            bins = Analysis.entropy_bins(s["entropy"])
            low, high = [], []
            for b in range(10):
                js, correct = s["js"][bins == b], s["correct"][bins == b]
                low.append(100 * correct[js <= np.median(js)].mean())
                high.append(100 * correct[js > np.median(js)].mean() if (js > np.median(js)).any() else np.nan)
            ax.plot(range(1, 11), low, marker="o", label="JS below decile median")
            ax.plot(range(1, 11), high, marker="o", label="JS above decile median")
            ax.set_title(domain)
            ax.set_xlabel("Entropy decile (1 = lowest entropy)")
            ax.set_ylabel("Zero-shot accuracy (%)")
            ax.set_ylim(0, 100)
            ax.legend()
        plt.tight_layout()

        path = Path(self.fig_dir, f"within_bin_{self.model_name}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Figure saved at {path}")

def main(model_name: str, frac: float) -> None:
    ana = Analysis(model_name=model_name, frac=frac)
    print(ana.signal_table().round(3).to_string(index=False))
    ana.plot_js_hist()
    ana.plot_within_bin()

if __name__ == "__main__":
    main("vitb16", 1.0)