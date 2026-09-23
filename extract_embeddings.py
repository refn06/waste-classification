"""
Extract frozen CLIP embeddings for the waste classification dataset.

Expected folder structure (this matches the Kaggle "Waste Classification Data"
dataset by techsash, but works for any ImageFolder-style layout):

    DATASET/
        TRAIN/
            O/   <- organik images
            R/   <- anorganik (recyclable) images
        TEST/
            O/
            R/

If you add a third class (e.g. B3), just add another subfolder — this script
does not hardcode class names, it reads them from the folder names.

Usage:
    python extract_embeddings.py \
        --data_dir DATASET \
        --out_dir embeddings \
        --model_name openai/clip-vit-large-patch14-336 \
        --batch_size 64
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def list_images(split_dir: Path):
    """Return (filepaths, labels, class_names) for an ImageFolder-style dir."""
    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    filepaths, labels = [], []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for idx, cls in enumerate(class_names):
        cls_dir = split_dir / cls
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in exts:
                filepaths.append(str(f))
                labels.append(idx)
    return filepaths, labels, class_names


@torch.no_grad()
def extract(filepaths, model, processor, device, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(filepaths), batch_size), desc="Extracting"):
        batch_paths = filepaths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[warn] skipping unreadable image {p}: {e}")
                images.append(Image.new("RGB", (336, 336)))
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model.get_image_features(**inputs)
        # transformers versions differ in what get_image_features() returns:
        # some give a plain tensor, some a ModelOutput with .image_embeds.
        # Handle both, with a manual fallback via the vision tower as last resort.
        if torch.is_tensor(outputs):
            feats = outputs
        elif hasattr(outputs, "image_embeds"):
            feats = outputs.image_embeds
        else:
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output
            feats = model.visual_projection(pooled_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
        embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path with TRAIN/ and TEST/ subfolders")
    ap.add_argument("--out_dir", default="embeddings")
    ap.add_argument("--model_name", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device}")
    print(f"[info] loading model: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_name)

    class_names = None
    for split in ["TRAIN", "TEST"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[warn] {split_dir} not found, skipping")
            continue

        filepaths, labels, classes = list_images(split_dir)
        if class_names is None:
            class_names = classes
        print(f"[info] {split}: {len(filepaths)} images across {len(classes)} classes: {classes}")

        feats = extract(filepaths, model, processor, device, args.batch_size)
        np.save(out_dir / f"{split.lower()}_features.npy", feats)
        np.save(out_dir / f"{split.lower()}_labels.npy", np.array(labels))

    with open(out_dir / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    print(f"[done] embeddings + labels saved to {out_dir}/")


if __name__ == "__main__":
    main()
