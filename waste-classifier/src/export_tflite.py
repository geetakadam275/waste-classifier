"""Convert the trained Keras model to TensorFlow Lite for edge devices.

Usage:
    python src/export_tflite.py --quantization float16   # or int8, none
"""
import argparse

import numpy as np
import tensorflow as tf

import config


def representative_dataset():
    ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=config.IMG_SIZE,
        batch_size=1,
        shuffle=True,
        seed=config.SEED,
        label_mode=None,
    )
    for batch in ds.take(100):
        yield [tf.cast(batch, tf.float32)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantization", choices=["none", "float16", "int8"], default="float16"
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(config.MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    config.TFLITE_PATH.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1e6
    print(f"Saved {config.TFLITE_PATH} ({size_mb:.2f} MB, quantization={args.quantization})")


if __name__ == "__main__":
    main()
