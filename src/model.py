
from __future__ import annotations

import tensorflow as tf


def compile_binary_classifier(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


def build_mobilenetv2_classifier(
    image_size: tuple[int, int] = (224, 224),
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="mammogram")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="malignant_score")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_mammography_classifier")

    return compile_binary_classifier(model, learning_rate)


def load_trained_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)
