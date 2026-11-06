import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import model_names, k_values

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

def main() -> None:
    summary = Summary(rate=0.5, gamma=16)
    df = summary.collect()
    print(summary.key_table(df).to_string(index=False))
    summary.plot_readout(df, rules=["P1", "B2"])

if __name__ == "__main__":
    main()