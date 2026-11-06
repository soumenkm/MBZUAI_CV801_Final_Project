import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from experiments import Experiments
from analysis import Analysis
from config import domains, model_names, k_values, order_seeds, alpha, beta

class Utility:
    def __init__(self, model_name: str, dataset: str, domain: str, k: int) -> None:
        """it measures how much each stored entry of a finished run helps on the heldout images,
        which were never part of any stream"""
        self.model_name = model_name
        self.dataset = dataset
        self.domain = domain
        self.k = k

        self.exp = Experiments(model_name=model_name, domain=domain, k=k, dataset=dataset)
        self.test = self.exp.features["test"]
        self.signals = self.exp.signals["test"]
        self.heldout = self.exp.load_features("heldout")

        self.table_dir = Path(Path.cwd(), "outputs", "tables", "utility", dataset, f"k{k}")
        self.table_dir.mkdir(parents=True, exist_ok=True)

    def load_entries(self, name: str, rate: float, order_seed: int) -> np.ndarray:
        """it returns the test indices of the images stored in the final cache of a run"""
        rate, gamma = Analysis.settings(name, rate, 0.0)
        with open(self.exp.run_path(name, rate, gamma, order_seed), mode="r", encoding="utf-8") as f:
            snapshots = json.load(f)["snapshots"]

        return np.array(snapshots[max(snapshots, key=int)])

    def leave_one_out(self, idx: np.ndarray) -> np.ndarray:
        """it returns, for each stored entry, how many points of heldout accuracy are lost when only that
        entry is removed. positive means the entry helps. the readout is the sum, as in TDA"""
        feats, labels = self.test["feats0"][idx], self.signals["pred"][idx]
        h_feats, h_logits, h_labels = self.heldout["feats0"], self.heldout["logits0"], self.heldout["labels"]

        weights = np.exp(-beta * (1 - h_feats @ feats.T))
        scores = h_logits + alpha * weights @ np.eye(h_logits.shape[1])[labels]
        top2 = np.argsort(-scores, axis=1)[:, :2]
        top1, second = top2[:, 0], top2[:, 1]
        second_score = scores[np.arange(len(scores)), second]
        correct = top1 == h_labels

        utility = np.zeros(len(idx))
        for j, c in enumerate(labels):
            # removing entry j lowers only the score of its label c, so only images predicted as c can change
            hit = top1 == c
            new_score = scores[hit, c] - alpha * weights[hit, j]
            new_pred = np.where(new_score >= second_score[hit], c, second[hit])
            utility[j] = 100 * (correct[hit].sum() - (new_pred == h_labels[hit]).sum()) / len(h_labels)

        return utility

    def utility_table(self, names: list = ["C0", "P1", "B2"], rate: float = 0.5, orders: list = order_seeds[:3]) -> pd.DataFrame:
        """it saves the utility of every stored entry and returns a summary per condition: how many entries
        help or hurt, and whether utility follows label correctness, entropy or JS"""
        rows = []
        for name in names:
            for order_seed in orders:
                idx = self.load_entries(name, rate, order_seed)
                utility = self.leave_one_out(idx)
                for i, u in zip(idx, utility):
                    rows.append({"condition": name, "order": order_seed, "idx": int(i), "utility": u,
                                 "label_correct": bool(self.signals["pred"][i] == self.test["labels"][i]),
                                 "entropy": float(self.signals["entropy"][i]), "js": float(self.signals["js"][i])})
        entries = pd.DataFrame(rows)
        entries.to_csv(Path(self.table_dir, f"entries_{self.model_name}_{self.domain.replace(' ', '_')}.csv"), index=False)

        summary = []
        for name, df in entries.groupby("condition", sort=False):
            summary.append({
                "domain": self.domain,
                "condition": name,
                "entries": len(df) / len(orders),
                "total_utility": df["utility"].sum() / len(orders),
                "helpful_share": 100 * (df["utility"] > 0).mean(),
                "harmful_share": 100 * (df["utility"] < 0).mean(),
                "utility_correct_label": df.loc[df["label_correct"], "utility"].mean(),
                "utility_wrong_label": df.loc[~df["label_correct"], "utility"].mean(),
                # low entropy and low JS should mean a useful entry, hence the minus sign
                "spearman_neg_entropy": spearmanr(df["utility"], -df["entropy"]).statistic,
                "spearman_neg_js": spearmanr(df["utility"], -df["js"]).statistic
            })

        return pd.DataFrame(summary)

def main(model_name: str, dataset: str, k: int) -> None:
    tables = [Utility(model_name, dataset, domain, k).utility_table() for domain in domains[dataset]]
    df = pd.concat(tables, ignore_index=True)

    path = Path(Path.cwd(), "outputs", "tables", "utility", dataset, f"k{k}", f"summary_{model_name}.csv")
    df.round(4).to_csv(path, index=False)
    print(f"{model_name}, {dataset}, k={k}: table saved at {path}")
    print(df.round(4).to_string(index=False))

if __name__ == "__main__":
    for model_name in model_names:
        for dataset in ["officehome", "imageneta", "imagenetr"]:
            for k in k_values:
                main(model_name, dataset, k)