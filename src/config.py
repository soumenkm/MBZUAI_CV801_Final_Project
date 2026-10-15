domains = ["Art", "Clipart", "Product", "Real World"]
split_fracs = {"dev": 0.1, "test": 0.7, "heldout": 0.2}

model_name_map = {"vitb16": "clip-vit-base-patch16", "vitb32": "clip-vit-base-patch32"}
template = "a photo of a {}."
logit_scale = 100.0

k = 3          # slots per class, TDA uses 3 for every dataset
alpha = 2.0    # TDA positive cache, ImageNet config
beta = 5.0