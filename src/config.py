domains = ["Art", "Clipart", "Product", "Real World"]
split_fracs = {"dev": 0.1, "test": 0.7, "heldout": 0.2}

model_name_map = {"vitb16": "clip-vit-base-patch16", "vitb32": "clip-vit-base-patch32"}
template = "a photo of a {}."
logit_scale = 100.0

k_values = [3, 8]          # slots per class, TDA uses 3 for every dataset
alpha = 2.0    # TDA positive cache, ImageNet config
beta = 5.0

target_rates = [0.25, 0.5, 0.75]
gamma_grid = [1, 4, 16, 64, 256]   # wide because JS is small for most images
order_seeds = list(range(10))
adaptive_lam = 0.05
# rule: who may enter, readout: how votes are added ("sum", "norm" or "mass")
conditions = {
    "B0": {"rule": "always", "readout": "sum", "use_gamma": False},
    "C0": {"rule": "always", "readout": "sum", "use_gamma": False},
    "P1": {"rule": "js", "readout": "sum", "use_gamma": False},
    "P2": {"rule": "always", "readout": "sum", "use_gamma": True},
    "P3": {"rule": "js", "readout": "sum", "use_gamma": True},
    "B2": {"rule": "mean_entropy", "readout": "sum", "use_gamma": False},
    "B3": {"rule": "entropy_of_mean", "readout": "sum", "use_gamma": False},
    "B4": {"rule": "random", "readout": "sum", "use_gamma": False},
    "B6": {"rule": "adaptive", "readout": "sum", "use_gamma": False},
    "BA": {"rule": "agree", "readout": "sum", "use_gamma": False},

    # same admission rules, but each class votes as if its bucket were full
    "C0n": {"rule": "always", "readout": "norm", "use_gamma": False},
    "P1n": {"rule": "js", "readout": "norm", "use_gamma": False},
    "B2n": {"rule": "mean_entropy", "readout": "norm", "use_gamma": False},
    "B3n": {"rule": "entropy_of_mean", "readout": "norm", "use_gamma": False},
    "B4n": {"rule": "random", "readout": "norm", "use_gamma": False},
    "B6n": {"rule": "adaptive", "readout": "norm", "use_gamma": False},

    # JS readout that keeps the total vote of the cache the same
    "P2m": {"rule": "always", "readout": "mass", "use_gamma": True}
}