import torch
import transformers
from pathlib import Path
from data import OfficeHomeDataset
from config import model_name_map, template, logit_scale

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

if __name__ == "__main__":
    main("vitb16", "Art", 64)
    main("vitb16", "Clipart", 64)