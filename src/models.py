import torch
import transformers
from pathlib import Path
from data import OfficeHomeDataset
from config import model_name_map, template, logit_scale
import numpy as np

class CLIPModel:
    def __init__(self, model_name: str) -> None:
        """model_name is either vitb16 or vitb32. the weights are kept in ./models"""
        if not torch.cuda.is_available():
            raise ValueError("Sorry, cuda is not available!")

        self.model_name = model_name
        self.device = "cuda"
        self.cwd = Path.cwd()
        self.model_path = Path(self.cwd, "models")
        self.model_path_full = Path(self.model_path, model_name_map[self.model_name])

        self.model = transformers.CLIPModel.from_pretrained(self.model_path_full, local_files_only=True).to(self.device)
        self.processor = transformers.CLIPProcessor.from_pretrained(self.model_path_full, local_files_only=True)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, class_names: list) -> torch.Tensor:
        """it encodes one prompt per class, for example 'a photo of a backpack.'
        and returns unit length text features of shape [num_classes, dim]"""
        prompts = [template.format(c) for c in class_names]
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to(self.device)
        # text encoder output followed by the projection into the shared image-text space
        outputs = self.model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_feats = self.model.text_projection(outputs.pooler_output).float()

        return text_feats / text_feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        """it encodes a list of PIL images and returns unit length image features of shape [num_images, dim]"""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        # image encoder output followed by the projection into the shared image-text space
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        image_feats = self.model.visual_projection(outputs.pooler_output).float()

        return image_feats / image_feats.norm(dim=-1, keepdim=True)

    @staticmethod
    def get_logits(image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """it returns the zero-shot class scores, which is cosine similarity times the logit scale"""
        return logit_scale * image_feats @ text_feats.T

    @staticmethod
    def get_probs(logits: torch.Tensor) -> np.ndarray:
        """it turns class scores into probabilities. float64 is used so that tiny probabilities don't become 0"""
        return torch.softmax(logits.double(), dim=-1).cpu().numpy()

    @staticmethod
    def entropy(probs: np.ndarray) -> np.ndarray:
        """it returns the entropy of each row. low entropy means the model is confident"""
        # 0 * log 0 is taken as 0, so log is computed only where prob > 0
        log_probs = np.log(np.where(probs > 0, probs, 1.0))

        return -(probs * log_probs).sum(axis=-1)

    @staticmethod
    def js_divergence(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """it returns how much the predictions on the original and the flipped image disagree.
        it is 0 when both are the same and at most ln 2 when they have no overlap"""
        mean = (p0 + p1) / 2
        # JS = entropy of the mean - mean of the two entropies. it is the same as the KL form but shorter
        js = CLIPModel.entropy(mean) - (CLIPModel.entropy(p0) + CLIPModel.entropy(p1)) / 2

        # tiny negative values can come from rounding
        return np.clip(js, 0, None)

    @staticmethod
    def get_signals(p0: np.ndarray, p1: np.ndarray) -> dict:
        """it computes everything the admission rules need from the two views.
        p0 is from the original image and p1 from the flipped one, both of shape [num_images, num_classes]"""
        signals = {
            "pred": p0.argmax(axis=-1),
            "entropy": CLIPModel.entropy(p0),
            "js": CLIPModel.js_divergence(p0, p1),
            "mean_entropy": (CLIPModel.entropy(p0) + CLIPModel.entropy(p1)) / 2,
            "entropy_of_mean": CLIPModel.entropy((p0 + p1) / 2),
            "agree": p0.argmax(axis=-1) == p1.argmax(axis=-1)
        }

        return signals

    @staticmethod
    def new_cache(num_classes: int, k: int, dim: int) -> dict:
        """it makes an empty cache with k slots per class. the slots of class c are rows c*k to (c+1)*k.
        an empty slot has label -1"""
        cache = {
            "num_classes": num_classes,
            "k": k,
            "feats": np.zeros((num_classes * k, dim), dtype=np.float32),
            "labels": np.full(num_classes * k, -1),
            "entropy": np.full(num_classes * k, np.inf),
            "js": np.zeros(num_classes * k),
            "idx": np.full(num_classes * k, -1)
        }

        return cache

    @staticmethod
    def add_to_cache(cache: dict, feat: np.ndarray, label: int, entropy: float, js: float, idx: int) -> str:
        """it tries to put an image into the slots of its predicted class.
        if there is an empty slot it is added. if the class is full, it replaces the least confident
        (highest entropy) entry, but only if the new image is more confident. it returns what happened"""
        start = label * cache["k"]
        empty = np.where(cache["labels"][start:start + cache["k"]] == -1)[0]
        if len(empty) > 0:
            pos = start + empty[0]
            action = "added"
        else:
            pos = start + cache["entropy"][start:start + cache["k"]].argmax()
            if entropy >= cache["entropy"][pos]:
                return "rejected"
            action = "replaced"

        cache["feats"][pos] = feat
        cache["labels"][pos] = label
        cache["entropy"][pos] = entropy
        cache["js"][pos] = js
        cache["idx"][pos] = idx

        return action

    @staticmethod
    def read_cache(cache: dict, feat: np.ndarray, beta: float, gamma: float = 0.0) -> np.ndarray:
        """it returns the vote of the cache for each class. entries that look like the image vote more.
        with gamma > 0, entries whose two views disagreed vote less. gamma = 0 is the ordinary cache"""
        votes = np.zeros(cache["num_classes"])
        filled = cache["labels"] >= 0
        if not filled.any():
            return votes

        sims = cache["feats"][filled] @ feat
        # similarity 1 gives weight 1, lower similarity gives a weight close to 0 (same as TDA and Tip-Adapter)
        weights = np.exp(-beta * (1 - sims))
        weights = weights * np.exp(-gamma * cache["js"][filled] / np.log(2))
        # add each entry's weight to the vote of its stored label
        np.add.at(votes, cache["labels"][filled], weights)

        return votes

def main(model_name: str, domain: str, n_images: int) -> None:
    model = CLIPModel(model_name=model_name)
    ds = OfficeHomeDataset(domain=domain, split="dev")
    text_feats = model.encode_text(ds.prompt_names())

    items = [ds[i] for i in ds.get_order(order_seed=0)[:n_images]]
    images = [OfficeHomeDataset._load_image(i["image_path"]) for i in items]
    flipped = [OfficeHomeDataset._flip_image(i) for i in images]
    labels = torch.tensor([i["label"] for i in items], device=model.device)

    pred_orig = CLIPModel.get_logits(model.encode_images(images), text_feats).argmax(dim=-1)
    pred_flip = CLIPModel.get_logits(model.encode_images(flipped), text_feats).argmax(dim=-1)
    print(f"{model_name}, {domain}: text feats {tuple(text_feats.shape)}, {n_images} images, logit scale in model: {model.model.logit_scale.exp().item():.1f}")
    print(f"accuracy original: {(pred_orig == labels).float().mean().item() * 100:.1f}%, flipped: {(pred_flip == labels).float().mean().item() * 100:.1f}%, views agree: {(pred_orig == pred_flip).float().mean().item() * 100:.1f}%")

    # checking JS on examples worked out by hand
    same = CLIPModel.js_divergence(np.array([[0.8, 0.2]]), np.array([[0.8, 0.2]]))[0]
    opposite = CLIPModel.js_divergence(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[0]
    close = CLIPModel.js_divergence(np.array([[0.8, 0.2]]), np.array([[0.6, 0.4]]))[0]
    print(f"JS same: {same:.4f} (expected 0), opposite: {opposite:.4f} (expected 0.6931), close: {close:.4f} (expected 0.0242)")

    p0 = CLIPModel.get_probs(CLIPModel.get_logits(model.encode_images(images), text_feats))
    p1 = CLIPModel.get_probs(CLIPModel.get_logits(model.encode_images(flipped), text_feats))
    signals = CLIPModel.get_signals(p0, p1)
    print(f"mean entropy: {signals['entropy'].mean():.3f}, mean JS: {signals['js'].mean():.4f}, JS < 0.01: {(signals['js'] < 0.01).mean() * 100:.1f}%, views agree: {signals['agree'].mean() * 100:.1f}%")

    # example: the query is [1, 0] and the three stored entries have similarity 0.95, 0.70 and 0.20 with it
    cache = CLIPModel.new_cache(num_classes=3, k=1, dim=2)
    for label, sim in [(0, 0.95), (1, 0.70), (2, 0.20)]:
        CLIPModel.add_to_cache(cache, np.array([sim, np.sqrt(1 - sim**2)]), label, entropy=0.1, js=0.0, idx=label)
    base = np.array([2.0, 2.1, 0.0])
    print(f"scores with cache: {base + CLIPModel.read_cache(cache, np.array([1.0, 0.0]), beta=5.0)}, expected [2.779 2.323 0.018]")

    # one class with 2 slots: 0.2 and 0.45 fill it, 0.3 replaces 0.45, 0.6 is rejected
    cache = CLIPModel.new_cache(num_classes=2, k=2, dim=2)
    actions = [CLIPModel.add_to_cache(cache, np.array([1.0, 0.0]), 0, entropy=e, js=0.0, idx=i) for i, e in enumerate([0.2, 0.45, 0.3, 0.6])]
    print(f"actions: {actions}, entropies kept: {cache['entropy'][:2]}, expected [0.2 0.3]")

    # with gamma = 2, an entry with JS = ln 2 should vote exp(-2) = 0.135 times as much as with gamma = 0
    cache["js"][:2] = np.log(2)
    print(f"vote with gamma 0: {CLIPModel.read_cache(cache, np.array([1.0, 0.0]), beta=5.0)[0]:.3f}, gamma 2: {CLIPModel.read_cache(cache, np.array([1.0, 0.0]), beta=5.0, gamma=2.0)[0]:.3f} (expected 2.000 and 0.271)")

if __name__ == "__main__":
    main("vitb16", "Art", 64)
    main("vitb16", "Clipart", 64)