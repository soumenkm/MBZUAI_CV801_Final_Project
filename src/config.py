data_paths = {"officehome": "data/OfficeHomeDataset", "imagenetr": "data/ImageNetR", "imageneta": "data/ImageNetA"}
domains = {"officehome": ["Art", "Clipart", "Product", "Real World"], "imagenetr": ["ImageNet-R"], "imageneta": ["ImageNet-A"]}
model_names = ["vitb16", "vitb32"]

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
evict_lam = 4.0   # E2 replaces by entropy + evict_lam * JS. fixed in advance, E1 already counts JS once
rescue_low = 0.5    # R1 admits every image below this dev entropy quantile
rescue_high = 0.8   # and images up to this quantile only if both views agree and JS is low
rescue_js = 0.5     # JS quantile among dev images between the two entropy quantiles

# rule: who may enter, readout: how votes are added ("sum", "norm", "mass" or "max"),
# evict: which value decides replacement (default "entropy"), score: what the readout weight uses (default "js")

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
    "P2m": {"rule": "always", "readout": "mass", "use_gamma": True},

    # each class votes only with its most similar entry
    "C0x": {"rule": "always", "readout": "max", "use_gamma": False},
    "P1x": {"rule": "js", "readout": "max", "use_gamma": False},
    "B2x": {"rule": "mean_entropy", "readout": "max", "use_gamma": False},
    "B3x": {"rule": "entropy_of_mean", "readout": "max", "use_gamma": False},
    "B4x": {"rule": "random", "readout": "max", "use_gamma": False},
    "B6x": {"rule": "adaptive", "readout": "max", "use_gamma": False},

    # admit all, but JS takes part in deciding which entry is replaced
    "E1": {"rule": "always", "readout": "sum", "evict": "entropy_of_mean", "use_gamma": False},
    "E2": {"rule": "always", "readout": "sum", "evict": "entropy_js", "use_gamma": False},

    # same weight values as JS readout, but given to entries at random within a class (P2s) or by entropy rank (P2h)
    "P2s": {"rule": "always", "readout": "sum", "score": "shuffled_js", "use_gamma": True},
    "P2h": {"rule": "always", "readout": "sum", "score": "entropy_ranked", "use_gamma": True},

    # JS used only for medium entropy images, where it carries some information
    "R1": {"rule": "rescue", "readout": "sum", "use_gamma": False},
    "R1x": {"rule": "rescue", "readout": "max", "use_gamma": False},

    # ordinary cache with the TDA order, so each image votes for itself
    "T0": {"rule": "always", "readout": "sum", "use_gamma": False, "insert_first": True},

    # E1 with the crop view instead of the flip, only the eviction key changes
    "E1c": {"rule": "always", "readout": "sum", "use_gamma": False, "evict": "entropy_of_mean", "view": "crop"},
}

# third view: a random resized crop, the same crop for an image in every run
crop_scale = (0.5, 1.0)
crop_seed = 42