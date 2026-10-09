"""Gradio web demo. Run: python app/app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr

from disposal import load_guide
from predict import load_model_and_labels, predict_image

model, labels = load_model_and_labels()
guide = load_guide()


def classify(image):
    if image is None:
        return {}, "Upload or capture an image."
    result = predict_image(image, model, labels, guide)
    verdict = "♻️ Recyclable" if result["recyclable"] else "🚫 Not recyclable"
    text = (
        f"**{result['label'].title()}** ({result['confidence'] * 100:.1f}%)\n\n"
        f"{verdict}\n\n**Bin:** {result['bin']}\n\n**Disposal tip:** {result['tip']}"
    )
    return result["probabilities"], text


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Waste item"),
    outputs=[gr.Label(num_top_classes=6, label="Class probabilities"), gr.Markdown()],
    title="Garbage / Waste Classifier",
    description="Identify the material, check if it is recyclable, and get disposal guidance.",
)

if __name__ == "__main__":
    demo.launch()
