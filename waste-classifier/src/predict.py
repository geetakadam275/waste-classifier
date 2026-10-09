"""Predict the waste class of a single image.

Usage:
    python src/predict.py path/to/image.jpg
"""
import argparse
import json

import numpy as np
import tensorflow as tf
from PIL import Image

import config
from disposal import get_guidance, load_guide


def load_model_and_labels():
    model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(config.IMG_SIZE)
    # Raw 0-255 pixels: the model applies MobileNetV2 preprocessing internally.
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def predict_image(image: Image.Image, model, labels, guide=None) -> dict:
    probs = model.predict(preprocess(image), verbose=0)[0]
    idx = int(np.argmax(probs))
    label = labels[idx]
    info = get_guidance(label, guide)
    return {
        "label": label,
        "confidence": float(probs[idx]),
        "probabilities": {l: float(p) for l, p in zip(labels, probs)},
        **info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    model, labels = load_model_and_labels()
    result = predict_image(Image.open(args.image), model, labels, load_guide())
    print(f"Item: {result['label']} ({result['confidence'] * 100:.1f}%)")
    print(f"Recyclable: {'YES' if result['recyclable'] else 'NO'}")
    print(f"Bin: {result['bin']}")
    print(f"Disposal: {result['tip']}")


if __name__ == "__main__":
    main()
