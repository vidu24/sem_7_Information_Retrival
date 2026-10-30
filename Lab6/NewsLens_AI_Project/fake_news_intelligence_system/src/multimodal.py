import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class ClipImageSearch:
    """Ranks a user-supplied image collection against a natural-language query."""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def search(self, query, images):
        if not images:
            return []
        inputs = self.processor(
            text=[query], images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits_per_text[0].softmax(dim=0).cpu().tolist()
        return sorted(
            [{"index": i, "score": float(score)} for i, score in enumerate(scores)],
            key=lambda x: x["score"], reverse=True
        )
