domains = ["Art", "Clipart", "Product", "Real World"]
split_fracs = {"dev": 0.1, "test": 0.7, "heldout": 0.2}

model_name_map = {"vitb16": "clip-vit-base-patch16", "vitb32": "clip-vit-base-patch32"}
template = "a photo of a {}."
logit_scale = 100.0

k = 3          # slots per class, TDA uses 3 for every dataset
alpha = 2.0    # TDA positive cache, ImageNet config
beta = 5.0

target_rates = [0.25, 0.5, 0.75]
gamma_grid = [1, 4, 16, 64, 256]   # wide because JS is small for most images
order_seeds = [0, 1, 2]
adaptive_lam = 0.05
conditions = {
    "B0": {"rule": "always", "use_gamma": False},
    "C0": {"rule": "always", "use_gamma": False},
    "P1": {"rule": "js", "use_gamma": False},
    "P2": {"rule": "always", "use_gamma": True},
    "P3": {"rule": "js", "use_gamma": True},
    "B2": {"rule": "mean_entropy", "use_gamma": False},
    "B3": {"rule": "entropy_of_mean", "use_gamma": False},
    "B4": {"rule": "random", "use_gamma": False},
    "B6": {"rule": "adaptive", "use_gamma": False},
    "BA": {"rule": "agree", "use_gamma": False}
}