"""
Gradio demo app for the waste classification model, for Hugging Face Spaces
(ZeroGPU hardware).

Files needed alongside this app.py in the Space:
    - requirements.txt
    - model_out/classifier.joblib
    - model_out/class_names.txt
"""

import spaces  # must be imported first, before torch / any CUDA-related package

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-large-patch14-336"
MODEL_DIR = Path("model_out")

LABEL_DISPLAY = {
    "O": "\U0001F966 Organik",
    "R": "\u267b\ufe0f Anorganik",
}

# Load everything on CPU at startup — ZeroGPU only attaches a GPU inside
# functions decorated with @spaces.GPU, so no .to("cuda") happens here.
print("[info] loading CLIP (CPU, startup)...")
clip_model = CLIPModel.from_pretrained(MODEL_NAME).eval()
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

class_names = (MODEL_DIR / "class_names.txt").read_text().strip().split("\n")
classifier = joblib.load(MODEL_DIR / "classifier.joblib")
print(f"[info] classes: {class_names}")


@spaces.GPU
@torch.no_grad()
def predict(image: Image.Image):
    if image is None:
        return {}, ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = clip_model.to(device)

    image = image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    outputs = model.get_image_features(**inputs)

    if torch.is_tensor(outputs):
        feats = outputs
    elif hasattr(outputs, "image_embeds"):
        feats = outputs.image_embeds
    else:
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        feats = model.visual_projection(vision_outputs.pooler_output)

    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.cpu().numpy()

    probs = classifier.predict_proba(feats_np)[0]
    result = {LABEL_DISPLAY.get(label, label): float(p) for label, p in zip(class_names, probs)}

    top_label = class_names[int(np.argmax(probs))]
    top_conf = float(np.max(probs))
    verdict = f"### {LABEL_DISPLAY.get(top_label, top_label)}\nConfidence: **{top_conf * 100:.1f}%**"

    return result, verdict


theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate",
)

CUSTOM_CSS = """
#header { text-align: center; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: var(--body-text-color-subdued); margin-bottom: 1.5rem; }
#verdict { text-align: center; padding: 1rem; }
.gradio-container { max-width: 900px !important; margin: auto; }
"""

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Klasifikasi Sampah") as demo:
    gr.Markdown("# \u267b\ufe0f Klasifikasi Sampah", elem_id="header")
    gr.Markdown(
        "Upload foto sampah untuk mengetahui kategorinya secara otomatis. "
        "Dikembangkan untuk **KOMPRES 16 Informatika — AI Innovation**.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload gambar sampah", height=320)
            submit_btn = gr.Button("Klasifikasikan", variant="primary", size="lg")
        with gr.Column(scale=1):
            verdict_output = gr.Markdown(elem_id="verdict")
            label_output = gr.Label(num_top_classes=2, label="Detail skor per kelas")

    gr.Markdown(
        "---\n"
        "**Model**: CLIP ViT-L/14@336 (frozen embedding) + Logistic Regression classifier "
        "\u00b7 Macro F1 pada test set: **0.98** \u00b7 5-fold CV: **0.975 \u00b1 0.001**"
    )

    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, verdict_output])
    image_input.change(fn=predict, inputs=image_input, outputs=[label_output, verdict_output])

if __name__ == "__main__":
    demo.launch()
