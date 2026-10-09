"""Train a MobileNetV2 transfer-learning waste classifier.

Usage:
    python src/train.py --epochs 10 --fine_tune_epochs 10
"""
import argparse
import json

import tensorflow as tf
from tensorflow.keras import layers, models

import config


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        class_names,
    )


def build_model(num_classes: int):
    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=config.IMG_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    args = parser.parse_args()

    config.MODELS_DIR.mkdir(exist_ok=True)
    train_ds, val_ds, class_names = load_datasets()
    print("Classes:", class_names)

    model, base = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
    ]

    # Stage 1: train the new classification head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Stage 2: fine-tune the top of the backbone
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[: config.FINE_TUNE_AT]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.FINE_TUNE_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=args.epochs,
            callbacks=callbacks,
        )

    model.save(config.MODEL_PATH)
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f)
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
