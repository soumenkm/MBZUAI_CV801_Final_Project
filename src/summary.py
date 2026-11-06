import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import model_names, k_values

from scipy.stats import spearmanr
from config import model_names, k_values, domains

class Summary:
    def __init__(self, rate: float = 0.5, gamma: float = 16) -> None:
        """it puts the interval tables of all backbones, datasets and K values together"""
        self.rate = rate
        self.gamma = gamma
        self.cwd = Path.cwd()
        self.output_dir = Path(self.cwd, "outputs")
        self.table_dir = Path(self.output_dir, "tables", "summary")
        self.fig_dir = Path(self.output_dir, "figures", "summary")
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.short = {"officehome": "OH", "imageneta": "IN-A", "imagenetr": "IN-R"}
        self.backbone = {"vitb16": "B/16", "vitb32": "B/32"}
        # all K = 3 settings first, then all K = 8 settings
        self.settings = [(m, d, k) for k in k_values for m in model_names for d in self.short]

    def collect(self) -> pd.DataFrame:
        """it reads the interval table of every setting written by analysis.py into one table"""
        tables = []
        for model_name, dataset, k in self.settings:
            path = Path(self.output_dir, "tables", dataset, f"k{k}", f"ci_{model_name}_rate{self.rate}_gamma{self.gamma}.csv")
            df = pd.read_csv(path)
            df.insert(0, "k", k)
            df.insert(0, "dataset", dataset)
            df.insert(0, "model", model_name)
            tables.append(df)
        df = pd.concat(tables, ignore_index=True)

        path = Path(self.table_dir, f"ci_all_rate{self.rate}_gamma{self.gamma}.csv")
        df.to_csv(path, index=False)
        print(f"Table saved at {path}")

        return df

    def key_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """it makes one row per setting with the main differences. a star marks an interval that excludes 0"""
        keys = [("P1", "C0"), ("P1x", "C0x"), ("B2", "C0"), ("B2x", "C0x"), ("C0x", "C0"), ("C0n", "C0"),
                ("T0", "C0"), ("E1", "C0"), ("P2", "C0")]
        rows = []
        for model_name, dataset, k in self.settings:
            row = {"setting": f"{self.short[dataset]} {self.backbone[model_name]} K={k}"}
            for name, ref in keys:
                r = df[(df["model"] == model_name) & (df["dataset"] == dataset) & (df["k"] == k) &
                       (df["condition"] == name) & (df["reference"] == ref)].iloc[0]
                row[f"{name} vs {ref}"] = f"{r['diff']:+.2f}" + ("*" if r["excludes_0"] else "")
            rows.append(row)
        table = pd.DataFrame(rows)

        path = Path(self.table_dir, f"key_rate{self.rate}_gamma{self.gamma}.csv")
        table.to_csv(path, index=False)
        print(f"Table saved at {path}")

        return table

    def plot_readout(self, df: pd.DataFrame, rules: list = ["P1", "B2"]) -> None:
        """for each setting it plots the gap of a selective rule to the cache that admits all images,
        under sum, norm and max readout, with 95% intervals. the gap under each readout is measured
        against the cache with the same readout"""
        labels = [f"{self.short[d]} {self.backbone[m]}" for m, d, k in self.settings]
        x = np.arange(len(labels))
        per_k = len(labels) // len(k_values)
        fig, axs = plt.subplots(len(rules), 1, figsize=(12, 3.2 * len(rules)), squeeze=False)
        for ax, rule in zip(axs[:, 0], rules):
            for j, (suffix, readout) in enumerate([("", "sum"), ("n", "norm"), ("x", "max")]):
                picked = [df[(df["model"] == m) & (df["dataset"] == d) & (df["k"] == k) &
                             (df["condition"] == rule + suffix)].iloc[0] for m, d, k in self.settings]
                y = np.array([p["diff"] for p in picked])
                err = [y - np.array([p["low"] for p in picked]), np.array([p["high"] for p in picked]) - y]
                ax.errorbar(x + (j - 1) * 0.25, y, yerr=err, fmt="o", capsize=2, label=f"{readout} readout")
            ax.axhline(0, color="gray", linestyle=":")

            # a line between the K groups and the name of each group at the bottom of the panel
            for i, k in enumerate(k_values):
                if i > 0:
                    ax.axvline(i * per_k - 0.5, color="black", linewidth=0.8)
                ax.text(i * per_k + (per_k - 1) / 2, -0.14, f"K = {k}", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=9, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("Difference (points)")
            ax.set_title(f"{rule} at {int(100 * self.rate)}% admission minus the cache that admits all images")
            ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()

        path = Path(self.fig_dir, f"readout_rate{self.rate}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Figure saved at {path}")

    def load_entries(self) -> pd.DataFrame:
        """it reads the entry utilities saved by utility.py for every setting and domain. the deciles and
        ranks are computed inside each cache, so caches of different size and datasets can be compared"""
        tables = []
        for model_name, dataset, k in self.settings:
            for domain in domains[dataset]:
                path = Path(self.output_dir, "tables", "utility", dataset, f"k{k}", f"entries_{model_name}_{domain.replace(' ', '_')}.csv")
                df = pd.read_csv(path)
                df["model"], df["dataset"], df["k"], df["domain"] = model_name, dataset, k, domain
                tables.append(df)
        df = pd.concat(tables, ignore_index=True)

        cache = ["model", "dataset", "k", "domain", "condition", "order"]
        for score in ["utility", "entropy", "js"]:
            df[f"{score}_rank"] = df.groupby(cache)[score].rank(pct=True)
        for score in ["entropy", "js"]:
            # 0 holds the tenth of the entries with the lowest score in their cache
            df[f"{score}_decile"] = np.minimum((10 * df[f"{score}_rank"]).astype(int), 9)

        return df

    def utility_corr(self, entries: pd.DataFrame, name: str = "C0", n: int = 1000) -> pd.DataFrame:
        """it returns the spearman correlation of entry utility with entropy and with JS, for each setting and
        for all settings together, with a 95% bootstrap interval over entries. positive means entries with
        high entropy (or high JS) help more"""
        entries = entries[entries["condition"] == name]
        groups = [(f"{self.short[d]} {self.backbone[m]} K={k}",
                   entries[(entries["model"] == m) & (entries["dataset"] == d) & (entries["k"] == k)])
                  for m, d, k in self.settings] + [("All", entries)]
        rng = np.random.default_rng(42)
        rows = []
        for setting, df in groups:
            row = {"setting": setting, "entries": len(df)}
            u = df["utility_rank"].to_numpy()
            for score in ["entropy", "js"]:
                s = df[f"{score}_rank"].to_numpy()
                values = []
                for _ in range(n):
                    idx = rng.integers(0, len(u), len(u))
                    values.append(spearmanr(u[idx], s[idx]).statistic)
                row[f"corr_{score}"] = spearmanr(u, s).statistic
                row[f"corr_{score}_low"], row[f"corr_{score}_high"] = np.quantile(values, [0.025, 0.975])
            rows.append(row)
        table = pd.DataFrame(rows)

        path = Path(self.table_dir, f"utility_corr_{name}.csv")
        table.round(4).to_csv(path, index=False)
        print(f"Table saved at {path}")

        return table

    def plot_utility(self, entries: pd.DataFrame, name: str = "C0") -> None:
        """it plots the mean utility of stored entries in each entropy decile (top) and JS decile (bottom)
        of their cache, one panel per dataset and one line per backbone and K"""
        entries = entries[entries["condition"] == name]
        fig, axs = plt.subplots(2, len(self.short), figsize=(4 * len(self.short), 6.5), squeeze=False)
        for row, score in enumerate(["entropy", "js"]):
            for col, dataset in enumerate(self.short):
                ax = axs[row][col]
                for model_name in model_names:
                    for k in k_values:
                        df = entries[(entries["model"] == model_name) & (entries["dataset"] == dataset) & (entries["k"] == k)]
                        means = df.groupby(f"{score}_decile")["utility"].mean()
                        ax.plot(means.index + 1, means.values, marker="o", label=f"{self.backbone[model_name]} K={k}")
                ax.axhline(0, color="gray", linestyle=":")
                ax.set_title(self.short[dataset])
                ax.set_xlabel(f"{'Entropy' if score == 'entropy' else 'JS'} decile of the entry in its cache (1 = lowest)")
                ax.set_ylabel("Mean utility (points)")
                ax.legend(fontsize=7)
        plt.tight_layout()

        path = Path(self.fig_dir, f"utility_{name}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Figure saved at {path}")

def main() -> None:
    summary = Summary(rate=0.5, gamma=16)
    # df = summary.collect()
    # print(summary.key_table(df).to_string(index=False))
    # summary.plot_readout(df, rules=["P1", "B2"])

    entries = summary.load_entries()
    print(summary.utility_corr(entries, name="C0").round(3).to_string(index=False))
    summary.plot_utility(entries, name="C0")

if __name__ == "__main__":
    main()