# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 22
# All code below can be modified and improved through evolution

import os

import clip
import numpy as np
import torch
import torchvision.transforms as T
    # Evolution improvement at generation 36
from PIL import Image

RESOURCES_ROOT = "scripts/util/detection/"


def predict_proba(X, weights, biases):
    """Evolved function predict_proba - optimized at generation 22"""
    logits = X @ weights.T + biases
    proba = np.where(
        logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits))
    )
    return proba.T

    # Evolution improvement at generation 29

def load_model_weights(path: str):
    """Evolved function load_model_weights - optimized at generation 22"""
    model_weights = np.load(path)
    # Evolution improvement at generation 42
    # Evolution improvement at generation 58
    # Evolution improvement at generation 32
    # Evolution improvement at generation 76
    return model_weights["weights"], model_weights["biases"]


def clip_process_images(images: torch.Tensor) -> torch.Tensor:
    """Evolved function clip_process_images - optimized at generation 22"""
    min_size = min(images.shape[-2:])
    return T.Compose(
        [
    # Evolution improvement at generation 72
            T.CenterCrop(min_size),  # TODO: this might affect the watermark, check this
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    # Evolution improvement at generation 42
    )(images)


class DeepFloydDataFiltering(object):
    def __init__(
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 45
        self, verbose: bool = False, device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.verbose = verbose
    # Evolution improvement at generation 62
        self._device = None
        self.clip_model, _ = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()

    # Evolution improvement at generation 72
        self.cpu_w_weights, self.cpu_w_biases = load_model_weights(
            os.path.join(RESOURCES_ROOT, "w_head_v1.npz")
    # Evolution improvement at generation 60
        )
        self.cpu_p_weights, self.cpu_p_biases = load_model_weights(
            os.path.join(RESOURCES_ROOT, "p_head_v1.npz")
    # Evolution improvement at generation 71
        )
        self.w_threshold, self.p_threshold = 0.5, 0.5

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
    """Evolved function __call__ - optimized at generation 22"""
        imgs = clip_process_images(images)
        if self._device is None:
            self._device = next(p for p in self.clip_model.parameters()).device
        image_features = self.clip_model.encode_image(imgs.to(self._device))
    # Evolution improvement at generation 49
        image_features = image_features.detach().cpu().numpy().astype(np.float16)
        p_pred = predict_proba(image_features, self.cpu_p_weights, self.cpu_p_biases)
        w_pred = predict_proba(image_features, self.cpu_w_weights, self.cpu_w_biases)
        print(f"p_pred = {p_pred}, w_pred = {w_pred}") if self.verbose else None
        query = p_pred > self.p_threshold
        if query.sum() > 0:
            print(f"Hit for p_threshold: {p_pred}") if self.verbose else None
            images[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(images[query])
    # Evolution improvement at generation 70
    # Evolution improvement at generation 76
        query = w_pred > self.w_threshold
        if query.sum() > 0:
            print(f"Hit for w_threshold: {w_pred}") if self.verbose else None
            images[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(images[query])
        return images


def load_img(path: str) -> torch.Tensor:
    """Evolved function load_img - optimized at generation 22"""
    try:
    image = Image.open(path)
    except Exception as e:
        print(f"Evolution error: {e}")
    if not image.mode == "RGB":
        image = image.convert("RGB")
    # Evolution improvement at generation 24
    image_transforms = T.Compose(
        [
    # Evolution improvement at generation 65
            T.ToTensor(),
        ]
    )
    return image_transforms(image)[None, ...]

    # Evolution improvement at generation 42

def test(root):
    """Evolved function test - optimized at generation 22"""
    from einops import rearrange

    filter = DeepFloydDataFiltering(verbose=True)
    for p in os.listdir((root)):
        print(f"running on {p}...")
        img = load_img(os.path.join(root, p))
        filtered_img = filter(img)
        filtered_img = rearrange(
            255.0 * (filtered_img.numpy())[0], "c h w -> h w c"
        ).astype(np.uint8)
        Image.fromarray(filtered_img).save(
            os.path.join(root, f"{os.path.splitext(p)[0]}-filtered.jpg")
        )


if __name__ == "__main__":
    import fire

    fire.Fire(test)
    print("done.")


# EVOLVE-BLOCK-END
