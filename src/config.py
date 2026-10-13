domains = ["Art", "Clipart", "Product", "Real World"]
split_fracs = {"dev": 0.1, "test": 0.7, "heldout": 0.2}

model_name_map = {"vitb16": "clip-vit-base-patch16", "vitb32": "clip-vit-base-patch32"}
template = "a photo of a {}."
logit_scale = 100.0