import numpy as np


class ClipEmbedder:
    """
    Lazy imports torch/open_clip to reduce import and packaging issues.
    Install extras:
      pip install "pixsorter[clip]"
    """
    def __init__(self, clip_model_key="vit_l_14", device=None):
        import torch
        import open_clip

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if clip_model_key == "vit_l_14":
            model_name, pretrained = "ViT-L-14", "openai"
        elif clip_model_key == "vit_b_32":
            model_name, pretrained = "ViT-B-32", "openai"
        else:
            raise ValueError("clip_model_key must be 'vit_l_14' or 'vit_b_32'")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        torch.set_grad_enabled(False)

    def embed_rgb(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage
        img = PILImage.fromarray(rgb)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)